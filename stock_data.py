import pandas as pd
import datetime
def get_stock_data(code, start_date, end_date):
    # 登陆系统
    bs.login()
    # 读取start_date至end_date期间的数据
    rs = bs.query_history_k_data_plus(
        code, "date,code,close,pctChg,volume", start_date=start_date, end_date=end_date, frequency="d", adjustflag="3")
    # 将读取信息格式化处理，更改为PandasDataFrame对象
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result.rename(columns={'date': "Date", 'close': "Close"}, inplace=True)
    result["Close"] = result.apply(lambda x: float(x['Close']), axis=1)
    result["Date"] = result.apply(
        lambda x: datetime.datetime.strptime(x['Date'], "%Y-%m-%d"), axis=1)
    result.set_index('Date', inplace=True)
    # 登出系统
    bs.logout()
    return result
