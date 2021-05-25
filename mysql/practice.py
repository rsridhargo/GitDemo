import pandas as pd
import openpyxl
with pd.ExcelWriter("test.xlsx",engine="openpyxl") as writer:
    book = openpyxl.load_workbook("test.xlsx")
    writer.book = book
    df = pd.read_csv("abc",header=0, index_col=False)
    writer.save()
