import pandas as pd


def test():
    df = pd.DataFrame({'A': ['1', '1'], 'B': [4, 5]})
    df = df.drop(columns=['A', 'C'])
    print(df.dtypes)

    # a = {'a':1, 'b':2}
    # print(a['c'])


def func1(row):
    return pd.Series((1,1))


if __name__ == "__main__":
    test()
