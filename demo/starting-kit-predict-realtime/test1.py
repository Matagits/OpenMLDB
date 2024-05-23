


def test():
    a = ['1', 2, 3]
    with open('list', "w") as fp:
        fp.write(str(a))

    with open('list', "r") as fp:
        b = eval(fp.read())

    c = 1



if __name__ == "__main__":
    test()
