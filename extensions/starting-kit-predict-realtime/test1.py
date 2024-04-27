import pandas as pd


def test():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df[['C', 'D']] = df.apply(func=func1, axis=1)
    print(df)
    df1 = df.iloc[0:2]
    print(df1)


def func1(row):
    return pd.Series((1,1))


if __name__ == "__main__":
    test()
