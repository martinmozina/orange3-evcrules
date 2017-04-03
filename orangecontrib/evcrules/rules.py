import numpy as np
import bottleneck as bn
from collections import defaultdict, Counter
from scipy.optimize import brentq
from scipy.special import gammaincc
from scipy.stats import chi2
from warnings import warn

from Orange.data import Table
from Orange.classification.rules import Rule, _RuleLearner, _RuleClassifier, \
    Evaluator, CN2UnorderedClassifier, get_dist, LRSValidator, Validator 
        

class RulesStar(_RuleLearner):
    """
    A rule learning algorithm that learns all rules in one pass without
    covering and separating. It uses a larger star of rules, therefore its
    memory consumption will be higher. On the other hand, it learns rules much 
    quicker than a standard separate and conquer algorithm, such as CN2. 
    """
    def __init__(self, preprocessors=None, base_rules=None, m=2, evc=True,
                 max_rule_length=5, width=100, default_alpha=1.0,
                 parent_alpha=1.0, add_sub_rules=False, target_class=None,
                 target_instances=None, classifier=CN2UnorderedClassifier):
        """
        Construct a rule learner.

        Parameters:
        -----------
        preprocessors and base_rules: The same as in _RuleLearner.
        m: The m parameter in m-estimate.
        evc: A boolean value specifying whether to use extreme value correction or not.
        max_rule_length: Maximum number of conditions in a rule.
        width: The width of star (beam) in the searching procedure.
        default_alpha: Required significance of rules calculated with LRS
        parent_alpha: Required significance of each condition in a rule (with 
            respect to the parent rule).
        add_sub_rules: Add all sub rules of best rules?
        target_class:
        """
        super().__init__(preprocessors, base_rules)
        self.rule_finder.search_strategy.evaluate = False

        # important to set evc first to initialize all components
        self.evaluator_norm = MEstimateEvaluator()
        self.evaluator_evc = EVCEvaluator()
        self.specialization_validator_norm = LRSValidator()
        self.rule_validator_norm = LRSValidator()
        self.specialization_validator_evc = EVCValidator()
        self.rule_validator_evc = EVCValidator()

        self.evc = evc
        self.default_alpha = default_alpha
        self.parent_alpha = parent_alpha
        self.m = m
        self.max_rule_length = max_rule_length
        self.width = width
        self.add_sub_rules = add_sub_rules
        self.evds = None
        self.target_class = target_class
        self.target_instances = target_instances  # learn rules for specific instances only
        self.store_intermediate_rules = False
        self.classifier = classifier


    def fit_storage(self, data):
        X, Y, W = data.X, data.Y, data.W if data.W else None
        Y = Y.astype(dtype=int)

        # estimate extreme value distributions (if necessary)
        if self.evc and not self.evds:
            warn("""Extreme value distributions not set.
                    Set evds property or run calculate_evds().
                    Returning an empty set of rules.""")
            return CN2UnorderedClassifier(domain=self.domain, rule_list=[])

        prior = get_dist(Y, W, self.domain)
        prior_prob = prior / prior.sum()
        # create initial star
        star = self.create_initial_star(X, Y, W, prior)
        # use visited to prevent learning the same rule all over again
        visited = set((r.covered_examples.tostring(), r.target_class) for r in star)
        # update best rule
        bestr = np.empty(X.shape[0], dtype=object)
        bestq = np.zeros(X.shape[0], dtype=float)
        for r in star:
            self.update_best(bestr, bestq, r, Y)
        # loop until star has rules
        self.inter_rules = [] # store intermediate rules
        while star:
            # specialize each rule in star
            new_star = []
            for r in star:
                #r.create_model()
                #print("s", r, r.curr_class_dist)
                # skip rules with pure distributions
                if r.curr_class_dist[r.target_class] == r.curr_class_dist.sum():
                    continue
                # refine rule
                rules = self.rule_finder.search_strategy.refine_rule(X, Y, W, r)
                # work refined rules
                for nr in rules:
                    nr.default_rule = nr.parent_rule.default_rule
                    nr.do_evaluate()
                    rkey = (nr.covered_examples.tostring(), nr.target_class)
                    if (rkey not in visited and
                            self.rule_finder.general_validator.validate_rule(nr) and
                            nr.quality >= nr.parent_rule.quality):
                        # rule is consistent with basic conditions
                        # can it be new best?
                        if self.rule_validator.validate_rule(nr):
                            self.update_best(bestr, bestq, nr, Y)
                        # can it be further specialized?
                        if (self.specialization_validator.validate_rule(nr) and
                                nr.length < self.max_rule_length):
                            new_star.append(nr)
                    visited.add(rkey)
            # assign a rank to each rule in new star
            nrules = len(new_star)
            inst_quality = np.zeros((X.shape[0], nrules))
            for ri, r in enumerate(new_star):
                if self.target_instances: # learn rules for specific instances only
                    c2 = np.zeros(r.covered_examples.shape, dtype = bool)
                    c2[self.target_instances] = 1
                    cov = np.where(c2 & r.covered_examples & (r.target_class == Y))[0]
                else:
                    cov = np.where(r.covered_examples & (r.target_class == Y))[0]
                inst_quality[cov, ri] = r.quality
            sel_rules = -(min(nrules, 5))
            queues = np.argsort(inst_quality)[:, sel_rules:]

            # create new star from queues
            new_star_set = set()
            index = -1
            while len(new_star_set) < self.width:
                if index < sel_rules:
                    break
                # pop one rule from each queue and put into a temporary counter
                cnt = Counter()
                for qi, q in enumerate(queues):
                    ri = q[index]
                    if inst_quality[qi, ri] > 0:
                        cnt[ri] += 1
                if not cnt:
                    break
                elts = cnt.most_common()
                for e, counts in elts:
                    if e in new_star_set: continue
                    new_star_set.add(e)
                    if len(new_star_set) >= self.width:
                        break
                index -= 1
            star = [new_star[ri] for ri in new_star_set]
            if self.store_intermediate_rules:
                rl = []
                vis = set()
                for ri, r in enumerate(bestr):
                    # add r
                    if r is None:
                        continue
                    self.add_rule(rl, vis, r)
                self.inter_rules.append(rl)

        # select best rules
        rule_list = []
        visited = set()
        for ri, r in enumerate(bestr):
            # add r
            if r is None:
                continue
            self.add_rule(rule_list, visited, r)
            if not hasattr(r, "best_instance"):
                r.best_instance = [data[ri]]
            else:
                r.best_instance.append(data[ri])
            if self.add_sub_rules:
                # create parent rule
                pr = self.create_parent(r, X, Y, W)
                while pr is not None:
                    # add parent rule
                    self.add_rule(rule_list, visited, pr)
                    pr = self.create_parent(pr, X, Y, W)
        rule_list = sorted(rule_list, key = lambda r: -r.quality)
        return self.classifier(domain=self.domain, rule_list=rule_list)

    def create_initial_star(self, X, Y, W, prior):
        star = []
        for cli, cls in enumerate(self.domain.class_var.values):
            if self.target_class is None or cli == self.target_class or cls == self.target_class:
                rules = self.rule_finder.search_strategy.initialise_rule(
                    X, Y, W, cli, self.base_rules, self.domain, prior, prior,
                    self.evaluator, self.rule_finder.complexity_evaluator,
                    self.rule_validator, self.rule_finder.general_validator)
                star.extend(rules)
        for r in star:
            r.default_rule = r
            r.do_evaluate()
        return star

    def create_parent(self, rule, X, Y, W):
        if rule.parent_rule:
            return rule.parent_rule
        if not rule.selectors:
            return None
        selectors = rule.selectors[:-1]
        new_rule = Rule(selectors=selectors,
                         domain=rule.domain,
                         initial_class_dist=rule.initial_class_dist,
                         prior_class_dist=rule.prior_class_dist,
                         quality_evaluator=rule.quality_evaluator,
                         complexity_evaluator=rule.complexity_evaluator,
                         significance_validator=rule.significance_validator,
                         general_validator=rule.general_validator)
        new_rule.filter_and_store(X, Y, W, rule.target_class)
        new_rule.do_evaluate()
        return new_rule

    def calculate_evds(self, data):
        evd_creator = EVDFitter(max_rule_length = self.max_rule_length)
        self.evds = evd_creator(data).evds

    @staticmethod
    def add_rule(rule_list, visited, rule):
        if rule.quality < rule.prior_class_dist[rule.target_class] / rule.prior_class_dist.sum() + 0.01:
            return
        rkey = (rule.covered_examples.tostring(), rule.target_class)
        if rkey not in visited:
            rule.create_model()
            rule_list.append(rule)
        visited.add(rkey)

    def update_best(self, bestr, bestq, rule, Y):
        indices = (rule.covered_examples) & (rule.target_class == Y) & \
                  (rule.quality-0.005 > bestq)
        bestr[indices] = rule
        bestq[indices] = rule.quality
    
    @property
    def m(self):
        return self._m

    @m.setter 
    def m(self, m):
        self._m = m
        self.evaluator_evc.m = m
        self.evaluator_norm.m = m

    @property
    def max_rule_length(self):
        return self._max_rule_length

    @max_rule_length.setter
    def max_rule_length(self, max_rule_length):
        self._max_rule_length = max_rule_length
        self.rule_finder.general_validator.max_rule_length = max_rule_length

    @property
    def evc(self):
        return self._evc

    @evc.setter
    def evc(self, evc):
        self._evc = evc
        if evc:
            self.evaluator = self.evaluator_evc
            self.specialization_validator = self.specialization_validator_evc
            self.rule_validator = self.rule_validator_evc
        else:
            self.evaluator = self.evaluator_norm
            self.specialization_validator = self.specialization_validator_norm
            self.rule_validator = self.rule_validator_norm

    @property
    def default_alpha(self):
        return self.rule_validator.default_alpha

    @default_alpha.setter
    def default_alpha(self, default_alpha):
        self.rule_validator_norm.default_alpha = default_alpha
        self.rule_validator_evc.default_alpha = default_alpha
    
    @property
    def parent_alpha(self):
        return self.specialization_validator.default_alpha

    @parent_alpha.setter
    def parent_alpha(self, parent_alpha):
        self.specialization_validator_norm.parent_alpha = parent_alpha
        self.specialization_validator_evc.parent_alpha = parent_alpha

    @property 
    def evds(self):
        return self._evds

    @evds.setter
    def evds(self, evds):
        self._evds = evds
        self.evaluator_evc.evds = evds
        self.specialization_validator_evc.evds = evds
        self.rule_validator_evc.evds = evds


def get_chi(p, n, P, N):
    """ Computes chi2 statisticts.

    Parameters:
    -----------
    p: covered positive examples
    n: covered negative examples
    P: all positive examples
    N: all negative examples
    """
    p = p - 0.5
    n = n + 0.5
    PN = P + N
    pn = p + n
    ep = pn*P/PN
    if p <= ep:
        return 0.0
    en = pn-ep
    if P == p:
        return 2 * (p*np.log(p/ep) + n*np.log(n/en))
    eP = (PN-pn)*P/PN
    eN = PN-pn-eP
    return 2 * (p*np.log(p/ep) + n*np.log(n/en) +
                (P-p)*np.log((P-p)/eP) + (N-n)*np.log((N-n)/eN))

class LRSEvaluator(Evaluator):
    def __init__(self, store=False):
        self.store = store
        self.reset_lrss()

    def reset_lrss(self):
        self.lrss = defaultdict(float)

    def evaluate_rule(self, rule):
        tc = rule.target_class
        dist = rule.curr_class_dist
        p_dist = rule.prior_class_dist
        # get consts
        n, N = dist.sum(), p_dist.sum()
        lrs = get_chi(dist[tc], n-dist[tc], p_dist[tc], N-p_dist[tc])
        if self.store: # store lrss for all rule lengths
            rl = rule.length
            self.lrss[rl] = max(self.lrss[rl], lrs)
            """self.lrss[(0, rl)] = max(self.lrss[(0, rl)], lrs)
            parent = rule.parent_rule
            while parent:
                p_dist = parent.curr_class_dist
                N = p_dist.sum()
                plrs = get_chi(dist[tc], n-dist[tc], p_dist[tc], N-p_dist[tc])
                self.lrss[(parent.length, rl)] = max(self.lrss[(parent.length, rl)], plrs)
                parent = parent.parent_rule"""
        return lrs

class MEstimateEvaluator(Evaluator):
    def __init__(self, m=2):
        self.m = m

    def evaluate_rule(self, rule):
        # as an exception, when target class is not set,
        # the majority class is chosen to stand against
        # all others
        tc = rule.target_class
        dist = rule.curr_class_dist
        if tc is None:
            tc = bn.nanargmax(dist)
        target = dist[tc]
        p_dist = rule.prior_class_dist
        pa = p_dist[tc] / p_dist.sum()
        return (target + self.m * pa) / (dist.sum() + self.m)

class EVCEvaluator(Evaluator):
    """ Evaluates a rule with the extreme value correction (EVC). """
    def __init__(self, m=2, evds=None):
        self.evds = evds

    def evaluate_rule(self, rule):
        # predicted class
        tc = rule.target_class

        # rule class distribution
        dist = rule.curr_class_dist
        dist_sum = dist.sum()
        acc = dist[tc]  / dist_sum
        # prior class distribution
        # TODO: default_rule has no influence anymore
        if hasattr(rule, "default_rule") and rule.default_rule.length>0:
            dr_length = rule.default_rule.length - 1 # rule.default_rule.length
        else:
            dr_length = 0
        p_dist = rule.prior_class_dist
        prior_sum = p_dist.sum()
        pa = p_dist[tc] / prior_sum

        # extreme value distribution
        if rule.length > len(self.evds[tc]):
            evd = self.evds[tc][-1]
        else:
            evd = self.evds[tc][rule.length-dr_length] #-dr_length]

        if evd.mu < 0.0001: # return as if rule distribution is not optimistic
            return (dist[tc] + self.m * pa) / (dist_sum + self.m)

        # compute chi
        n, N = dist.sum(), p_dist.sum()
        chi = get_chi(dist[tc], n-dist[tc], p_dist[tc], N-p_dist[tc])

        # corrected estimate
        if (evd.mu-chi)/evd.beta < -200 or acc <= pa:
            ePos = dist[tc]
        elif chi <= evd.median:
            ePos = pa * dist_sum
        else:
            diff = LNLNChiSq(evd, chi)
            try:
                corr_chi = brentq(diff, 0, chi, xtol=0.1, maxiter=20)
            except RuntimeError:
                corr_chi = 0
            if chi > 0:
                diff = LRInv(dist_sum, p_dist[tc], prior_sum, corr_chi)
                try:
                    ePos = brentq(diff, pa*dist_sum, dist[tc], xtol=0.1, maxiter=20)
                except RuntimeError:
                    ePos = pa * dist_sum
            else:
                ePos = pa * dist_sum
        q = (ePos + self.m * pa) / (dist_sum + self.m)
        # special case: when argument is evaluated low
        if hasattr(rule, "default_rule") and rule.default_rule.length>0 and \
                rule.default_rule.length == rule.length and q < pa+0.01:
            return min(pa + 0.01, acc)
        if q > pa:
            return q
        if acc < pa:
            return acc - 0.01
        return pa - 0.01 + 0.01*chi/evd.median

class EVCValidator(Validator):
    def __init__(self, default_alpha=1.0, parent_alpha=1.0, evds=None):
        self.parent_alpha = 1.0
        self.default_alpha = 1.0
        self.evds = evds

    def validate_rule(self, rule):
        tc = rule.target_class
        # rule class distribution
        dist = rule.curr_class_dist

        if self.default_alpha < 1.0:
            # TODO: fix that, default_rule has no influence any more
            if hasattr(rule, "default_rule") and rule.default_rule.length > 0:
                dr_length = rule.default_rule.length - 1 # rule.default_rule.length
            else:
                dr_length = 0
            # prior class distribution
            p_dist = rule.prior_class_dist
            if rule.length > len(self.evds[tc]):
                evd = self.evds[tc][-1]
            else:
                evd = self.evds[tc][rule.length-dr_length] #-dr_length]
            p = dist[tc]
            n = dist.sum() - p
            P = p_dist[tc]
            N = p_dist.sum() - P
            chi = get_chi(p, n, P, N)
            if evd.get_prob(chi) > self.default_alpha:
                return False

        if self.parent_alpha < 1.0 and rule.parent_rule is not None:
            evd = self.evds[tc][1]
            p = dist[tc]
            n = dist.sum() - p
            P = rule.parent_rule.curr_class_dist[tc]
            N = rule.parent_rule.curr_class_dist.sum() - P
            chi = get_chi(p, n, P, N)
            if evd.get_prob(chi) > self.parent_alpha:
                return False

        return True
 
class LNLNChiSq:
    def __init__(self, evd, chi):
        self.exponent = (evd.mu-chi)/evd.beta

    def __call__(self, chix):
        if chix > 1400:
            return -1000
        if chix <= 1e-6:
            chip = 0.5
        else:
            chip = gammaincc(0.5, chix/2)*0.5
        if chip <= 0.0:
            return -1000
        if chip < 1e-6:
            return np.log(chip)-self.exponent
        return np.log(-np.log(1-chip))-self.exponent
 
class LRInv:
    def __init__(self, n, P, N, corr_chi):
        self.n = n
        self.P = P
        self.N = N-P
        self.corr_chi = corr_chi

    def __call__(self, p):
        return get_chi(p, self.n-p, self.P, self.N) - self.corr_chi

class EVDDist:
    def __init__(self, mu, beta, median):
        self.mu = mu
        self.beta = beta
        self.median = median

    def get_prob(self, chi):
        if self.beta < 1.1: # use standard chi2 with 1 degree of freedom
            return 1.0-chi2.cdf(chi, 1)
        return 1.0-np.exp(-np.exp((self.mu-chi)/self.beta))

    def __str__(self):
        return "mu: {:.4f} beta: {:.4f} median: {:.4f}".format(self.mu, self.beta, self.median)

class EVDFitterClassifier(_RuleClassifier):
    def __init__(self, evds, domain):
        self.evds = evds
        self.domain = domain

class LengthValidator(Validator):
    def __init__(self, max_rule_length=5):
        self.max_rule_length = max_rule_length

    def validate_rule(self, rule):
        return rule.length <= self.max_rule_length

class PureRulesValidator(Validator):
    def validate_rule(self, rule):
        return rule.curr_class_dist[rule.target_class] == rule.curr_class_dist.sum()


class EVDFitter(RulesStar):
    def __init__(self, preprocessors=None, base_rules=None, n=30, seed=1,
                 max_rule_length=5):
        super().__init__(preprocessors, base_rules)
        self.evaluator_norm = LRSEvaluator()
        self.evc = False
        self.n = n
        self.seed = seed
        self.width = min(10, self.width)
        self.max_rule_length = max_rule_length
        self.store_intermediate_rules = True
        self.rule_finder.general_validator = LengthValidator(self.max_rule_length)

    def fit_storage(self, data): #X, Y, W=None):
        X, Y, W = data.X, data.Y, data.W if data.W else None
        Y = Y.astype(dtype=int)
        np.random.seed(self.seed)
        evd = {}

        for cls in range(len(self.domain.class_var.values)):
            self.target_class = cls
            print("estimating evd for class", cls)
            # repeat n-times
            max_vals = defaultdict(list)
            for i in range(self.n):
                print("{}/{}".format(i, self.n))
                # randomize class
                Yr = np.array(Y)
                np.random.shuffle(Yr)
                # learn rules
                new_data = Table.from_table(data.domain, data)
                new_data.Y = Yr
                super().fit_storage(new_data)
                for k in range(self.max_rule_length):
                    ki = k if k < len(self.inter_rules) else -1
                    max_vals[k+1].extend([r.quality for r in self.inter_rules[ki]])
            # calculate extreme value distributions
            evd_cls = {0:EVDDist(0, 1, 0)}
            for k in range(1, self.max_rule_length+1):
                median = bn.median(max_vals[k])
                mu = median + 2*np.log(np.log(2))
                if mu > 0.1:
                    evd_cls[k] = EVDDist(mu, 2, median)
                else:
                    evd_cls[k] = EVDDist(0, 1, 0)
            evd[cls] = evd_cls
            print()

        # returns an empty classifier
        return EVDFitterClassifier(evd, self.domain)

if __name__ == "__main__":
    data = Table('titanic')
    learner = RulesStar(evc=True)
    learner.calculate_evds(data)
    classifier = learner(data)

    for rule in classifier.rule_list:
        print(rule.curr_class_dist.tolist(), rule, rule.quality)
    print()

    data = Table('iris')
    learner = RulesStar(evc=True, parent_alpha=0.5)
    learner.rule_finder.general_validator.max_rule_length = 3
    learner.rule_finder.general_validator.min_covered_examples = 5
    learner.calculate_evds(data)
    classifier = learner(data)
    for rule in classifier.rule_list:
        print(rule, rule.curr_class_dist.tolist(), rule.quality)
    print()
