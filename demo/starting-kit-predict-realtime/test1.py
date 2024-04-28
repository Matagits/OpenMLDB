import pandas as pd


def test():
    df = pd.DataFrame({'A': [1, 1], 'B': [4, 5]})
    df1 = pd.DataFrame({'A': [1], 'C': [6]})
    df2 = pd.merge(df1, df, on='A', how='left', left_index=True)
    print(df2)


def func1(row):
    return pd.Series((1,1))


if __name__ == "__main__":
    test()
