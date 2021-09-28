import urllib
import pandas as pd
import json


class translator:

    def __init__(self,id,secret):
        # CLI
        self.client_id = id  # 개발자센터에서 발급받은 Client ID 값
        self.client_secret = secret  # 개발자센터에서 발급받은 Client Secret 값

    def trans_str(self,src,tar,txt):
        encText = urllib.parse.quote(txt)
        data = "source={}&target={}&text=".format(src, tar) + encText
        url = "https://openapi.naver.com/v1/papago/n2mt"

        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", self.client_id)
        request.add_header("X-Naver-Client-Secret", self.client_secret)
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
        response_body = response.read()
        return response_body.decode('utf-8')

    def trans_df_addcol(self,df,col,src,tar):

        newColname = str(col) + '_trans'

        df[newColname] = df[col].apply(
            lambda x: json.loads(
                self.trans_str(src, tar,x))['message']['result']['translatedText']
        )

        pass
