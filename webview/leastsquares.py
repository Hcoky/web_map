from sklearn.linear_model import LinearRegression  # 导入训练的模型，sklearn自带
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import folium
import webbrowser
import os
import pandas as pd
import seaborn as sns
import folium
import webbrowser
from folium.plugins import HeatMap


class webpic:

    def __init__(self, x, y, train):
        self.x = x
        self.y = y
        self.train = train

    def mat_linchart(self, x, y, color, edgecolor, pcolor, length, width, train):
        y = np.array(y)
        x = np.array(x)
        x = x.reshape(len(x), 1)
        y = y.reshape(len(y), 1)

        model = LinearRegression()  # 导入模型线性回归
        model.fit(x, y)  # 开始训练

        y = y.ravel()
        train = np.array(train)
        train.reshape(len(train), 1)
        y = np.append(y, train)

        y = y.reshape(len(y), 1)
        x = range(0, len(y))

        if color is None:
            color = 'orange'
        if edgecolor is None:
            edgecolor = 'black'
        if pcolor is None:
            pcolor = 'cornflowerblue'

        plt.figure(figsize=(width, length))
        plt.scatter(x, y, c=color, edgecolors=edgecolor)
        plt.plot(x, y, c=pcolor)
        # plt.show()
        # plt.savefig
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue())  # 将图片转为base64
        figdata_str = str(figdata_png, "utf-8")  # 提取base64的字符串，不然是b'xxx'

        # 保存为.html
        html = '<img src=\"data:image/png;base64,{}\"/>'.format(figdata_str)
        # plt.show()
        return html

    def bar(self, x, y, length, width, bar_width, color, edgecolor, line):
        plt.figure(figsize=(length, width), dpi=80)
        plt.bar(x, y, color=color, edgecolor=edgecolor, width=bar_width)
        if line:
            plt.scatter(x, y, c='orange', edgecolors='black')
            plt.plot(x, y, c="cornflowerblue")
            plt.plot(x, y)

            figfile = BytesIO()
            plt.savefig(figfile, format='png')
            figfile.seek(0)
            figdata_png = base64.b64encode(figfile.getvalue())  # 将图片转为base64
            figdata_str = str(figdata_png, "utf-8")  # 提取base64的字符串，不然是b'xxx'

            # 保存为.html
            html = '<img src=\"data:image/png;base64,{}\"/>'.format(figdata_str)
            return html

        # plt.show()

    def map(self, lon, lat, zoom_start, openwb):  # lon为经度，lat为纬度，zoom_start为地图缩放比例，openwb为True打开网页
        m = folium.Map(location=[lat, lon], zoom_start=zoom_start)
        m.add_child(folium.LatLngPopup())
        m.save("dimao.html")
        if openwb:
            webbrowser.open("dimao.html")

    def caloric(self, lat, lin, zoo_start, openwb):  # 用法与map()一样
        if lat is None:
            lat = 35
        if lin is None:
            lin = 110
        if zoo_start is None:
            zoo_start = 5

        path = os.path.split(os.path.realpath(__file__))[0]

        filename = "people.xlsx"

        posi = pd.read_excel(filename)

        num = 10
        #
        lat = np.array(posi["lat"][0:num])  # 获取维度之维度值
        # # print(lat)
        #
        lon = np.array(posi["lon"][0:num])  # 获取经度值
        pop = np.array(posi["pop"][0:num], dtype=float)  # 获取人口数，转化为numpy浮点型
        gdp = np.array(posi["GDP"][0:num], dtype=float)  # 获取人口数，转化为numpy浮点型
        #
        data1 = [[lat[i], lon[i], pop[i]] for i in range(num)]  # 将数据制作成[lats,lons,weights]的形式
        data1 = np.array(data1)
        print(data1.shape)

        lat = 30
        map_osm = folium.Map(location=[lat, lin], zoom_start=5)  # 绘制Map，开始缩放程度是5倍
        HeatMap(data1).add_to(map_osm)  # 将热力图添加到前面建立的map里

        file_path = (path + r'\人口.html')
        map_osm.save(file_path)  # 保存为html文件
        if openwb is True:
            webbrowser.open(file_path)


if __name__ == '__main__':
    a = webpic.mat_linchart(self=None, x=[1, 2, 3, 4, 5], y=[4, 5, 6, 9, 7],color='orange',edgecolor='black',pcolor='cornflowerblue',length=10, width=5,train=[10])
    """
    函数调用格式‘mat_linchart(self=None, x=[list], y=[list],length=int, width=int,train=[list])’
    x为x轴的坐标，y为y轴的坐标,length,width可以调整图的大小，train=[未来x的坐标]可预测未来y的值
    """

    # webpic.bar(self=None, x=[1, 2, 3], y=[4, 5, 6], length=6, width=6, bar_width=0.5, color="lightblue", edgecolor="cyan",
    # line=True)

    # webpic.map(self=None,lon=30,lat=116,zoom_start=5,openwb=True)
    #
    # webpic.caloric(self=None, lat=30, lin=116, zoo_start=6,openwb=True)
