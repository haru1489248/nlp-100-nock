import pandas as pd

df = pd.DataFrame({
    '名前': ['斎藤', '砂糖', '鈴木'],
    '年齢': [21, 30, 18],
})

sr = pd.Series(
    ['佐藤', '斎藤', '鈴木']
)


csv_df = pd.read_csv('tutrials/assets/dict.csv')
print(csv_df)

json_df = pd.read_json('tutrials/assets/sample.json')
print(json_df)

excel_df = pd.read_excel('tutrials/assets/user_data.xlsx') # sheet_nameはエクセルのシート名を指定してシートを読み込むことができる
print(excel_df.loc[[0, 3], ['ユーザID', '住所']]) # locはcolumn名で指定する
print(excel_df[['ユーザID', '住所']]) # column 名のリストを返す
print(excel_df.iloc[[1, 2], [0, 3]]) # ilocはindex番号で指定する

print(excel_df[excel_df['年齢']>= 20]) # 年齢が20歳以上の人を抜き出す
print(excel_df[(excel_df['年齢']>=20)&(excel_df['住所']=='埼玉県')])

print(excel_df['年齢'].mean(numeric_only=True)) # meanは平均, numeric_onlyは数値のみの平均を取得するモード。基本的にはコレを明示しないといけない

excel_df['新ポイント'] = excel_df['レベル']*excel_df['ポイント'] # 新しくカラムを作成する方法
print(excel_df['新ポイント'])

df = pd.read_csv('tutrials/assets/sale.csv')
print(df)
print(df.groupby('担当者').max(numeric_only=True)) # 担当者でグループ化して最大値を取得する
print(df.groupby('担当者').mean(numeric_only=True)) # 担当者でグループ化して平均値を取得する

index = range(1,5) # 1から4を返す（使うまで実行されない）
print(list(index))

df.index = range(1,10) # indexを振り直すことができる(元は0から始まっていたものを1から始まるようにした)
print(df)

df_new = df.set_index('日付') # columnをindexにすることができる
print(df_new)

print([str(x).zfill(3) for x in range(1, 10)]) # 001から009まで作成したリストを返す
df.index = [str(x).zfill(3) for x in range(1, 10)]
print(df)

print(df.columns)
df.columns = ['date', 'day of week','name', 'sales']
print(df)

df.set_index('name', inplace=True) # inplace true にするとdf自体が変わる（破壊的変更）
df_new = df.reset_index() # 上から順番にインデックスを貼り直す（inplace trueにするとdf自体が変更される）,元々のindexは列として追加される
df_new = df.reset_index(drop=True) # drop trueにすると元のindexを列に追加しない

df_1 = pd.read_csv('tutrials/assets/concat1.csv')
df_2 = pd.read_csv('tutrials/assets/concat2.csv')

df_3 = pd.concat([df_1, df_2]) # 縦にdfをつなげる
print(df_3)

df_4 = pd.concat([df_1, df_2], axis=1) # 横にdfをつなげる
print(df_4)
