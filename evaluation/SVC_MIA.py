import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.utils import shuffle

# when p > 0: computes p * p.log(); when p <= 0: substitutes a 0
# entropy quantifies the uncertainty in the model's predictions.
# higher entropy indicates more uncertainty in the predictions.
def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

# considers both model's predicted probabilities and the actual class labels
# help assess how the model's confidence in its predictions, especially when model is wrong
def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model):
    if data_loader is None:
        return torch.zeros([0, 10]), torch.zeros([0])

    prob = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            try:
                batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
                data, target = batch
            except:
                device = (
                    torch.device("cuda:0")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                data, target = get_x_y_from_data_dict(batch, device)
            with torch.no_grad():
                output = model(data)
                prob.append(F.softmax(output, dim=-1).data)
                targets.append(target)
                # print('output', output)

    return torch.cat(prob), torch.cat(targets)

"""
(Support Vector Machine)
Trains an SVM using shadow data and tests its performance on target data. 
The function returns the mean accuracy of predictions on target data.
"""
def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    # calculates the number of samples in each dataset
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]


    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    # shadow training data: label 1    the shadow testing data: label 0
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    X_shadow, Y_shadow = shuffle(X_shadow, Y_shadow)

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    # print('X_shadow', len(X_shadow), len(Y_shadow))
    clf.fit(X_shadow, Y_shadow)

    accs = []

    if n_target_train > 0:  # if the target training dataset is non-empty
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        # acc_train: proportion of samples predicted as 1 (i.e.as part of the training set)
        accs.append(acc_train)

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        # acc_test: proportion of samples predicted as 0 (i.e.as part of the test set)
        accs.append(acc_test)

    return np.mean(accs)

def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def SVC_MIA(shadow_train, target_train, target_test, shadow_test, model):
    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model) #shadow_train_prob: (10000,10) 10000 samples, 10 classes
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model)

    target_train_prob, target_train_labels = collect_prob(target_train, model)
    target_test_prob, target_test_labels = collect_prob(target_test, model)

    shadow_train_corr = (   # (10000,1) consist of 0 and 1, 1: predicted label by model matches the actual label
        torch.argmax(shadow_train_prob, axis=1) == shadow_train_labels
    ).int()
    shadow_test_corr = (
        torch.argmax(shadow_test_prob, axis=1) == shadow_test_labels
    ).int()
    target_train_corr = (
        torch.argmax(target_train_prob, axis=1) == target_train_labels
    ).int()
    target_test_corr = (
        torch.argmax(target_test_prob, axis=1) == target_test_labels
    ).int()

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None]) # (10000,1) model's predicted probability of the ground-truth label
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
    target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])

    # print('shadow_train_prob', shadow_train_prob)
    # print('shadow_train_conf', shadow_train_conf)

    shadow_train_entr = entropy(shadow_train_prob) # (10000,1) entropy of the model's predicted probability distribution
    shadow_test_entr = entropy(shadow_test_prob)

    target_train_entr = entropy(target_train_prob)
    target_test_entr = entropy(target_test_prob)

    shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
    shadow_test_m_entr = m_entropy(shadow_test_prob, shadow_test_labels)
    if target_train is not None:
        target_train_m_entr = m_entropy(target_train_prob, target_train_labels)
    else:
        target_train_m_entr = target_train_entr
    if target_test is not None:
        target_test_m_entr = m_entropy(target_test_prob, target_test_labels)
    else:
        target_test_m_entr = target_test_entr

    # acc_corr = SVC_fit_predict(
    #     shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr
    # )
    acc_conf = SVC_fit_predict(
        shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf
    )
    # acc_entr = SVC_fit_predict(
    #     shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr
    # )
    # acc_m_entr = SVC_fit_predict(
    #     shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr
    # )
    # acc_prob = SVC_fit_predict(
    #     shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob
    # )
    m = {
        # "correctness": acc_corr,
        "confidence": acc_conf,
        # "entropy": acc_entr,
        # "m_entropy": acc_m_entr,
        # "prob": acc_prob,
    }
    return m
