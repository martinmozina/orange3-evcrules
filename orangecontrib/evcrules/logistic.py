from Orange.data import Table
import rules

class LogisticRegression:

if __name__ == "__main__":
    data = Table('titanic')
    learner = rules.RulesStar(evc=True)
    classifier = learner(data)

    for rule in classifier.rule_list:
        print(rule.curr_class_dist.tolist(), rule, rule.quality)
    print()

    data = Table('iris')
    learner = rules.RulesStar(evc=True, parent_alpha=0.5)
    learner.rule_finder.general_validator.max_rule_length = 3
    learner.rule_finder.general_validator.min_covered_examples = 5
    classifier = learner(data)
    for rule in classifier.rule_list:
        print(rule, rule.curr_class_dist.tolist(), rule.quality)
    print()