from Orange.data import Table
from Orange.evaluation import CrossValidation, CA, AUC, LogLoss
from Orange.classification import LogisticRegressionLearner, NaiveBayesLearner, \
                                  RandomForestLearner
import orangecontrib.evcrules.rules as rules
from orangecontrib.evcrules.logistic import LRRulesLearner

datasets = ['ionosphere', 'adult_sample', 'iris', 'ionosphere', 'breast-cancer', 'bupa', 'titanic']
for d in datasets:
    data = Table(d)
    rule_learner = rules.RulesStar(evc=True, add_sub_rules=True)
    rule_learner_m = rules.RulesStar(evc=False, m=22, add_sub_rules=True)
    # compare lr with rules, lr without rules and sklearn's lr
    learners = [LRRulesLearner(opt_penalty=True, rule_learner=rule_learner),
                LRRulesLearner(opt_penalty=True, rule_learner=rule_learner_m),
                LRRulesLearner(opt_penalty=True), 
                LogisticRegressionLearner(C=1),
                NaiveBayesLearner(),
                RandomForestLearner()]
    res = CrossValidation(data, learners, k=5)
    print("Dataset: ", d)
    for l, ca, auc, ll in zip(learners, CA(res), AUC(res), LogLoss(res)):
        print("learner: {}\nCA: {}\nAUC: {}\n LL: {}".format(l, ca, auc, ll))

