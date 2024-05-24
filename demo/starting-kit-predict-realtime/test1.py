class A:
    def __init__(self):
        self.a = 3

    def run(self):
        print("A run")

    def test(self):
        self.test1()
        print("A test")

    def test1(self):
        pass


class B(A):
    def __init__(self):
        super().__init__()
        self.b = 0

    def run(self):
        # super().run()
        print("B run")

    def test1(self):
        print("test1")


def test():
    b = B()
    print(b.a)
    b.test()
    a = A()
    a.test()



if __name__ == "__main__":
    test()
