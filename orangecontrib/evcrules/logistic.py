import numpy as np
import scipy.optimize as opt

from Orange.data import Table
from Orange.classification import Learner, Model
from Orange.preprocess import (RemoveNaNClasses, RemoveNaNColumns,
                               Impute, Normalize)

import orangecontrib.evcrules.rules as rules


class LogisticRegression(Learner):
    """
    An implementation of L2-penalized logistic regression. If a rule
    learner is provided, learned rules get encoded as binary
    attributes (0 - not covered, 1 - covered) that are then
    appended to the original set of attributes. The extended set of
    attributes is used in the logistic regression model. To prevent
    overfitting, the weights of new attributes are penalized more (see
    Možina et al. Extreme value correction in rule learning)
    """
    preprocessors = [RemoveNaNClasses(),
                     RemoveNaNColumns(),
                     Impute(),
                     Normalize()]

    def __init__(self, preprocessors=None, penalty=1, rule_learner=None,
                 fit_intercept=True, intercept_scaling=1):
        """
        Parameters
        ----------
        preprocessors : list of Preprocess
            A sequence of data preprocessors to apply on data prior to
            fitting the model.
        pen : L2-penalty in loss function.
        rule_learner: Rule learner used to construct new attributes.
        """
        super().__init__(preprocessors)
        self.penalty = penalty
        self.rule_learner = rule_learner
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling

    def fit_storage(self, data):
        X, Y, W = data.X, data.Y, data.W if data.W else None

        # learn rules
        if self.rule_learner:
            rules = self.rule_learner(data).rule_list
        else:
            rules = []

        # add rules to X
        Xr = np.concatenate([X]+[r.covered_examples[:, np.newaxis] for r in rules],
                            axis=1)
        # add constant to X
        if self.fit_intercept:
            Xr = self.add_intercept(self.intercept_scaling, Xr)

        # set additional penalties for rules
        gamma = [0] * X.shape[1]
        for r in rules:
            gamma.append(r.curr_class_dist[r.target_class] - r.quality * r.curr_class_dist.sum())
        if self.fit_intercept:
            gamma.append(0)
        gamma = np.array(gamma)

        # build model
        w = []
        if len(self.domain.class_var.values) > 2:
            for cli, cls in enumerate(self.domain.class_var.values):
                # create class with domain {-1, 1}
                yc = np.ones_like(Y)
                yc[Y != cli] = -1
                # set bounds
                bounds = self.set_bounds(X, rules, cli)
                w.append(self.fit_params(Xr, yc, bounds, gamma))
        else:
            yc = np.ones_like(Y)
            yc[Y != 0] = -1
            bounds = self.set_bounds(X, rules, 0)
            x = self.fit_params(Xr, yc, bounds, gamma)
            w.append(x)
            w.append(-x)
        return LogisticClassifier(w, rules, self.fit_intercept,
                                  self.intercept_scaling, self.domain)

    @staticmethod
    def add_intercept(intercept, X):
        return np.hstack((X, intercept * np.ones((X.shape[0], 1))))

    def set_bounds(self, X, rules, cli):
        bounds = [(None, None) for _ in range(X.shape[1])]
        for r in rules:
            if r.target_class == cli:
                bounds.append((0, None))
            else:
                bounds.append((None, 0))
        if self.fit_intercept:
            bounds.append([None, None])
        return bounds

    def fit_params(self, X, y, bounds, gamma):
        w0 = np.zeros(X.shape[1])
        out = opt.minimize(self.ll, w0, args=(X, y, gamma), method='TNC',
                           bounds=bounds, jac=self.gradient)
        return out.x

    def phi(self, t):
        # logistic function, returns 1 / (1 + exp(-t))
        idx = t > 0
        out = np.empty(t.size, dtype=np.float)
        out[idx] = 1. / (1 + np.exp(-t[idx]))
        exp_t = np.exp(t[~idx])
        out[~idx] = exp_t / (1. + exp_t)
        return out

    def ll(self, w, X, y, gamma):
        # loss function to be optimized, it's the logistic loss
        z = X.dot(w)
        yz = y * z
        idx = yz > 0
        out = np.zeros_like(yz)
        out[idx] = np.log(1 + np.exp(-yz[idx]))
        out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
        out = out.sum()
        # add penalty
        out += (self.penalty * .5 * w).dot(w)
        # add second penalty (which is lasso-like and is a numpy array)
        out += gamma.dot(np.abs(w))
        return out

    def gradient(self, w, X, y, gamma):
        # gradient of the logistic loss (ll)
        z = X.dot(w)
        z = self.phi(y * z)
        z0 = (z - 1) * y
        gradll = X.T.dot(z0)
        # add penalties
        gradll += self.penalty * w
        # second penalty
        pos = w > 0
        neg = w < 0
        gradll[pos] += gamma[pos]
        gradll[neg] -= gamma[neg]
        return gradll

class LogisticClassifier(Model):
    def __init__(self, w, rules, fit_intercept,
                 intercept_scaling, domain):
        super().__init__(domain)
        self.w = w
        self.rules = rules
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling


if __name__ == "__main__":
    data = Table('breast-cancer')
    #data = Table('iris')
    rule_learner = rules.RulesStar(evc=True)
    learner = LogisticRegression(rule_learner = rule_learner)
    classifier = learner(data)

    #for rule in classifier.rule_list:
    #    print(rule.curr_class_dist.tolist(), rule, rule.quality)
    #print()

    #data = Table('iris')
    #learner = rules.RulesStar(evc=True, parent_alpha=0.5)
    #learner.rule_finder.general_validator.max_rule_length = 3
    #learner.rule_finder.general_validator.min_covered_examples = 5
    #classifier = learner(data)
    #for rule in classifier.rule_list:
    #    print(rule, rule.curr_class_dist.tolist(), rule.quality)
    #print()