import pandas as pd

books_df = pd.read_csv('/Users/salomonmuhirwa/Desktop/book r system/books/Books.csv', dtype={'Year-Of-Publication': str})
users_df = pd.read_csv('/Users/salomonmuhirwa/Desktop/book r system/books/Users.csv')

print(books_df.head())

