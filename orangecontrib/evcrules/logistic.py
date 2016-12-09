import numpy as np
import scipy.optimize as opt

from Orange.data import Table, Domain
from Orange.classification import Learner, Model
from Orange.evaluation import CrossValidation, LogLoss
from Orange.preprocess import (RemoveNaNClasses, RemoveNaNColumns,
                               Impute, Normalize, Continuize)
import orangecontrib.evcrules.rules as rules

class LRRulesLearner(Learner):
    """
    Learner learns a set of rules by using the provided
    rule learner. Then, learned rules get encoded as binary attributes
    (0 - not covered, 1 - covered) and together with the original set of
    attributes comprise the set of attributes used in logistic regression
    learning.

    If rule_learner is not provided, this acts as an ordinary
    logistic regression.

    The fitter for logistic regression uses the L2-penalized loss function.
    To prevent overfitting due to attributes built from rules,
    the weights of new rule-based attributes are penalized more (see
    Možina et al. Extreme value correction in rule learning)

    TODO: weights are not supported yet.
    """
    name = 'logreg rules'
    preprocessors = [RemoveNaNClasses(),
                     RemoveNaNColumns(),
                     Impute()]

    def __init__(self, preprocessors=None, penalty=1, opt_penalty=False,
                 rule_learner=None, basic_attributes=True,
                 fit_intercept=True, intercept_scaling=2):
        """
        Parameters
        ----------
        preprocessors :
            A sequence of data preprocessors to apply on data prior to
            fitting the model.
        penalty : L2-penalty in loss function.
        rule_learner: Rule learner used to construct new attributes.
        fit_intercept: Should we add a constant column to data?
        intercept_scaling: Value of constant in the intercept column. Note that
            intercept column is appended after normalization, therefore higher
            values will be less affected by penalization.
        """
        super().__init__(preprocessors)
        self.penalty = penalty
        self.opt_penalty = opt_penalty
        self.rule_learner = rule_learner
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.basic_attributes = basic_attributes
        # Post rule learning preprocessing should not decrease the
        # number of examples.
        self.post_rule_preprocess = [Normalize(), Continuize()]

    def fit_storage(self, data):
        if self.opt_penalty:
            self.penalty = self.tune_penalty(data)
        # learn rules
        rules = self.rule_learner(data).rule_list if self.rule_learner else []
        # preprocess data
        if not self.basic_attributes:
            domain = Domain([], data.domain.class_vars, data.domain.metas)
            data = data.from_table(domain, data)
        for pp in self.post_rule_preprocess:
            data = pp(data)
        # create data
        X, Y, W = data.X, data.Y, data.W if data.W else None
        # 1. add rules to X
        Xr = np.concatenate([X]+[r.covered_examples[:, np.newaxis] for r in rules],
                            axis=1)
        # 2. add constant to X
        if self.fit_intercept:
            Xr = self.add_intercept(self.intercept_scaling, Xr)
        # set additional penalties that penalized rule-based attributes
        gamma = self.get_gamma(X, rules)
        # build model
        w = []
        se = []
        if len(self.domain.class_var.values) > 2:
            for cli, cls in enumerate(self.domain.class_var.values):
                # create class with domain {-1, 1}
                yc = np.ones_like(Y)
                yc[Y != cli] = -1
                # set bounds
                bounds = self.set_bounds(X, rules, cli)
                x, s = self.fit_params(Xr, yc, bounds, gamma)
                w.append(x)
                se.append(s)
        else:
            yc = np.ones_like(Y)
            yc[Y != 0] = -1
            bounds = self.set_bounds(X, rules, 0)
            x, s = self.fit_params(Xr, yc, bounds, gamma)
            w = [x, -x]
            se = [s, s]
        # remove zero weights and corresponding rules
        to_keep, final_rules = list(range(X.shape[1])), []
        for ri, r in enumerate(rules):
            if any(wi[X.shape[1]+ri] != 0 for wi in w):
                to_keep.append(X.shape[1]+ri)
                final_rules.append(r)
        if self.fit_intercept:
            to_keep.append(-1)
        w = [wi[to_keep] for wi in w]
        se = [s[to_keep] for s in se]
        return LRRulesClassifier(w, se, final_rules, self.fit_intercept,
                                 self.intercept_scaling, self.domain, data.domain)

    def tune_penalty(self, data):
        learner = LRRulesLearner(fit_intercept=self.fit_intercept,
                                 intercept_scaling=self.intercept_scaling)
        penalties = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 100.]
        scores = []
        for pen in penalties:
            learner.penalty = pen
            res = CrossValidation(data, [learner], k=5, random_state=1111)
            ll = LogLoss(res)
            scores.append(ll)
        return penalties[scores.index(min(scores))]

    def get_gamma(self, X, rules):
        gamma = [0] * X.shape[1]
        for r in rules:
            gamma.append(r.curr_class_dist[r.target_class] -
                         r.quality * r.curr_class_dist.sum())
        if self.fit_intercept:
            gamma.append(0)
        return np.array(gamma)


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
            bounds.append((None, None))
        return bounds

    def fit_params(self, X, y, bounds, gamma):
        w0 = np.zeros(X.shape[1])
        out = opt.minimize(self.ll, w0, args=(X, y, gamma), method='TNC',
                           bounds=bounds, jac=self.gradient)
        w = out.x
        # compute standard errors (s)
        z = self.phi(X.dot(w))
        weights = z * (1 - z)
        xwx = (X.T * weights).dot(X)
        diag = np.diag_indices(X.shape[1])
        xwx[diag] += self.penalty
        inv = np.linalg.inv(xwx)
        s = inv[diag]
        return w, s

    @staticmethod
    def phi(t):
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

class LRRulesClassifier(Model):
    def __init__(self, w, se, rules, fit_intercept, intercept_scaling,
                 domain, postrule_domain):
        super().__init__(domain)
        self.w = np.array(w)
        self.se = np.array(se)
        self.rule_list = rules
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.postrule_domain = postrule_domain

    def predict_storage(self, data):
        data_post = data.from_table(self.postrule_domain, data)
        Xr = np.concatenate([data_post.X]+[r.evaluate_data(data.X)[:, np.newaxis] for r in self.rule_list],
                            axis=1)
        if self.fit_intercept:
            Xr = LRRulesLearner.add_intercept(self.intercept_scaling, Xr)
        probs = self.predict_proba(Xr)
        values = probs.argmax(axis=1)
        return values, probs

    def predict_proba(self, X):
        ps = np.empty((X.shape[0], len(self.domain.class_var.values)),
                      dtype=np.float)
        for i, cl in enumerate(self.domain.class_var.values):
            z = X.dot(self.w[i])
            ps[:,i] = LRRulesLearner.phi(z)
        ps = ps / np.linalg.norm(ps, ord=1, axis=1)[:,np.newaxis]
        return ps

    def __str__(self):
        desc = ""
        names = [at.name for at in self.postrule_domain.attributes] + \
                [str(r) for r in self.rule_list]
        if self.fit_intercept:
            names.append('intercept({})'.format(self.intercept_scaling))
        desc += '\t'.join(['{}={}'.format(self.domain.class_var.name, cl)
                           for i, cl in enumerate(self.domain.class_var.values)]) + \
                '\t' + 'Attribute' + '\n'
        for ni, n in enumerate(names):
            desc += '\t'.join(['{:.3}({:.3},{:.3})'.format(self.w[i,ni],
                                                           self.w[i,ni]-1.96*self.se[i,ni],
                                                           self.w[i,ni]+1.96*self.se[i,ni])
                               for i, cl in enumerate(self.domain.class_var.values)]) + \
                    '\t' + n + '\n'
        return desc


if __name__ == "__main__":
    data = Table('titanic')
    # learn a logistic regression model without rules
    lr = LRRulesLearner()(data)

    # learn a logistic regression using rules and basic attributes
    rule_learner = rules.RulesStar(evc=True, add_sub_rules=True)
    lr = LRRulesLearner(rule_learner=rule_learner)(data)
    print("Relevant rules when basic attributes are used: ")
    for rule in lr.rule_list:
        print(rule, rule.curr_class_dist.tolist(), rule.quality)

    # learn a logistic regression using rules and basic attributes
    rule_learner = rules.RulesStar(evc=True, add_sub_rules=True)
    lr = LRRulesLearner(rule_learner=rule_learner, basic_attributes=False)(data)
    print("Relevant rules without basic attributes:")
    for rule in lr.rule_list:
        print(rule, rule.curr_class_dist.tolist(), rule.quality)
