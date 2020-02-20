
class DSRule(object):
    """
    Wrapper for labeled lambdas, used to print rules in DSModel
    """

    def __init__(self, ld, caption=""):
        """
        Creates a new DSRule
        :param ld: Predicate of the rule (X->bool). Given an instance determines if the rule is aplicable
        :param caption: Description of the rule
        """
        self.ld = ld
        self.caption = caption

    def __str__(self):
        return self.caption

    def __call__(self, *args, **kwargs):
        return self.ld(*args, **kwargs)


if __name__ == '__main__':
    r1 = DSRule(lambda x: x > 3, "x greater than 3")
    assert str(r1) == "x greater than 3"
    assert r1(5)
    assert not r1(2)
