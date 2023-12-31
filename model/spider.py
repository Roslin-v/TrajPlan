import requests
import time
import parsel
import csv
import json


# 在去哪儿网站上获取景点的建议游玩时间
def get_recommend_time():
    with open('./data/recommend_time.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['name', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # 写入表头
    for page in range(1, 57):
        print(f'=========正在爬取第{page}页数据内容=========')
        time.sleep(2)
        url = f'https://travel.qunar.com/search/place/22-xiamen-299782/4-----0/{page}'
        # 请求头:把python代码伪装成浏览器 给服务器发送请求
        headers = {
            'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36 Edg/112.0.1722.58',
            'Cookie': 'SECKEY_ABVK=xjHYNyvTER514p26YGWC343OQQWweJehu66r/6cPc24%3D; BMAP_SECKEY=y0qt7sYdJfhqzvqZ18l67KmoJlBuqetWD1zdz5MzwuZrQHvAwF6UUZ9zuKNRvz9pQEvygHkfFMg7RTqqxusNqVCSbZh9q_jHzbKvreaWNS2Ki_fgaM9PUa-LxFRWBpYjj2b_AS5EMFSJvegKEMYmSOA5ll-No9gP4F-vAHnGvhb5l17PnvU8aVvLT2o30cj-; QN1=00008c002eb4517eb5a80697; QN99=6026; QunarGlobal=10.66.70.17_7b56d759_18879a3b151_-210e|1685672123197; qunar-assist={%22version%22:%2220211215173359.925%22%2C%22show%22:false%2C%22audio%22:false%2C%22speed%22:%22middle%22%2C%22zomm%22:1%2C%22cursor%22:false%2C%22pointer%22:false%2C%22bigtext%22:false%2C%22overead%22:false%2C%22readscreen%22:false%2C%22theme%22:%22default%22}; csrfToken=5jYhreufGA78TXFn4YaTat5MVYZwDgya; QN601=5cfdfde462b48b806337ef3c5d223618; _i=VInJOmiPqzxq2ZR1YnO78qZyXWtq; QN163=0; QN269=55D9685000EB11EE9B7CFA163E8F381D; fid=c24aeb7c-c5a0-486d-8072-0817e281a724; QN48=tc_9c1923fcdb9ffb69_18879e3687e_3940; ariaDefaultTheme=null; QN243=44; QN57=16856739703210.8698434013169207; Hm_lvt_15577700f8ecddb1a927813c81166ade=1685673970; QN63=%E6%B2%B3%E6%BA%90; QN300=s%3Dbing; QN71=MTEzLjgxLjMuMjEyOuaDoOW3njox; QN205=s%3Dbing; QN277=s%3Dbing; _vi=mWJI1m2_O13zAn2BVaJgsh9oiy2HENM6-UJhhjMeQBCgf0pG6Cr0bvzKWAB1JpR-DS_3uI0Y31Yt4bRAjNKFdDRAaRIDfLq_1SB027_HYVW1AD-NL9v6blu0jF2dElT7idV_5VbLsXD6vpFmW6Knzr3d7iy7KsytuU6mHXT11v_m; QN267=085053407291d013c7; QN58=1685759689485%7C1685763244022%7C12; JSESSIONID=22D7903E31288266A0301A5F89AADD87; __qt=v1%7CVTJGc2RHVmtYMStoSGxOT3A4RzZGN05DNllpRzlqMm1kM0x5VXAxdGFJdUh4SXZJZVZZOUFrdjA1cmhSeFUwd3lVbDA4UWVrb3J4ZDhZbytYcDdMM1NWT2hpNXJ0SmYwb1dnb3lNMjNhWXY1R0pKM21kUlE5MW1GOGorZ1RMbjI5OTgzZ0NQaVlPZVdHOXdQRmxnN1FtRXZFQWFBLzByZjRKNXNpclppV1ljPQ%3D%3D%7C1685763244171%7CVTJGc2RHVmtYMS9IM2FHam5MNUpEVFEySngvV3NZYWtQaTlUNGRMNThSQ3VQRUo5cGFSTkorTUlhWGVvUUlzQUQwNmJTK21haFh4M01tYXRDSFZ1Umc9PQ%3D%3D%7CVTJGc2RHVmtYMSsvNjh2VTBDVmNTY3F0WE9CTWdPUWtCcFFNcDFLdjJrSXB0cUhLSGVJR09lbXBiaVczRlhTcGQyKzRXbkpTek1XTC9oRlZoM2xYMng4SVVMRGZlaHJrU3VtRkdUdUx5WGpDN0U2NWVERWVKenkrOTNGUURaSy9EeWVpcnQrUXlZVk5mY3NiaDFpdzVSTnRmWTRwUEkxN2lmdlI5cW5KcEp5Q2RiT1YxQ1BSSEM2MFZSZDNuLy96T3kwYklIcm9uOWpqZnI3bTBMWW93ZWZyMmEyQ21HUW5NbHJWYVR5Q3VIMXNjSzNTTXFYWUpLTG5HdVh0YnBueUk0VDlHSWpENWxyTnRFays0NytlS1ZyQmxrTmdwbEo5OFhGbktWOEFCK2IvbEpiZG4rTzFxWlgrQ2pmMUl6ZkdMeEErNUVyVjBEaHhWN1AzMC9Nd0ZQK2xWelNUVUgyNW0rTUVlQTVKbGlhR0tST29oRzRhempOZVVESG1LMDlTdWlWQXpHWnErQXk3YmdCQWQwZG81dkhoc1NsQW9YaVJBOWlQZjN5MEpsRzJtS2gyeXY2QzE0OHArbDFoNGlVcW5MWGFkM080SjErck56bnZsTi9FcjNZOWtSMUNHYS9Dd0QrQ09LY3RpNmZCSE9GUWFwV3NaUzZQRGZOakxLMHhRTmo1ZUNhSko3ZDZEWXdHeWdpYURDUVpmenJtK1hwUjFLdU04UFFBdUNKVlR5VlRaYlpsMU9BZmpqd3lkT2paQzJlQUVjM0J2cG1FKzlYcXlQUVVjbVc3UkJ1b1U3RU1BVHVQemJzR0FiU2hCZXNWam9FTjJqa2IrSVQ0Q3h4SGVleHlOVDRYT2R3K2hOYzc1TEZIQ3lPSWRLM1NNcG5mUWJkRVNiNTZxZ0ZwZnZtZEhzNFBKWk5hWHJ6MDI4aGhuYWhrWDFjdDQ4S3Z0aEpqMjVSenZVaVdQMjJ3VzBvNUhoenpNTVVhU0dvPQ%3D%3D; Hm_lpvt_15577700f8ecddb1a927813c81166ade=1685763244; QN271=83df3432-fa03-46bb-a8e5-df1a3312d171',
        }
        response = requests.get(url=url, headers=headers)
        selector = parsel.Selector(response.text)
        ul = selector.css('.b_destlist')
        lis = ul.css('.e_info')
        # 模式a代表末行追加
        with open('./data/recommend_time.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['name', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for li in lis:
                name = li.css('.d_tit .tit').get()
                recommend_time = li.css('.d_days').get()
                index1 = name.rfind('</span>')
                index2 = name.rfind(('</a>'))
                name = name[index1+7:index2]
                if recommend_time is not None:
                    recommend_time = recommend_time[27:-6]
                dit = {
                    'name': name,
                    'time': recommend_time,
                }
                writer.writerow(dit)


# 在去哪儿网站上获取游记，作为用户轨迹
def get_traj():
    with open('./data/traj.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['user', 'day', 'seq', 'name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # 写入表头
    user = 0    # 用户id编号
    for page in range(1, 201):
        print(f'=========正在爬取第{page}页数据内容=========')
        time.sleep(2)
        url = f'https://travel.qunar.com/search/gonglue/22-xiamen-299782/start_heat/{page}.htm'
        # 请求头:把python代码伪装成浏览器 给服务器发送请求
        headers = {
            'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Mobile Safari/537.36 Edg/112.0.1722.58',
            'Cookie': 'SECKEY_ABVK=xjHYNyvTER514p26YGWC343OQQWweJehu66r/6cPc24%3D; BMAP_SECKEY=y0qt7sYdJfhqzvqZ18l67KmoJlBuqetWD1zdz5MzwuZrQHvAwF6UUZ9zuKNRvz9pQEvygHkfFMg7RTqqxusNqVCSbZh9q_jHzbKvreaWNS2Ki_fgaM9PUa-LxFRWBpYjj2b_AS5EMFSJvegKEMYmSOA5ll-No9gP4F-vAHnGvhb5l17PnvU8aVvLT2o30cj-; QN1=00008c002eb4517eb5a80697; QN99=6026; QunarGlobal=10.66.70.17_7b56d759_18879a3b151_-210e|1685672123197; qunar-assist={%22version%22:%2220211215173359.925%22%2C%22show%22:false%2C%22audio%22:false%2C%22speed%22:%22middle%22%2C%22zomm%22:1%2C%22cursor%22:false%2C%22pointer%22:false%2C%22bigtext%22:false%2C%22overead%22:false%2C%22readscreen%22:false%2C%22theme%22:%22default%22}; csrfToken=5jYhreufGA78TXFn4YaTat5MVYZwDgya; QN601=5cfdfde462b48b806337ef3c5d223618; _i=VInJOmiPqzxq2ZR1YnO78qZyXWtq; QN163=0; QN269=55D9685000EB11EE9B7CFA163E8F381D; fid=c24aeb7c-c5a0-486d-8072-0817e281a724; QN48=tc_9c1923fcdb9ffb69_18879e3687e_3940; ariaDefaultTheme=null; QN243=44; QN57=16856739703210.8698434013169207; Hm_lvt_15577700f8ecddb1a927813c81166ade=1685673970; QN63=%E6%B2%B3%E6%BA%90; QN300=s%3Dbing; QN71=MTEzLjgxLjMuMjEyOuaDoOW3njox; QN205=s%3Dbing; QN277=s%3Dbing; _vi=mWJI1m2_O13zAn2BVaJgsh9oiy2HENM6-UJhhjMeQBCgf0pG6Cr0bvzKWAB1JpR-DS_3uI0Y31Yt4bRAjNKFdDRAaRIDfLq_1SB027_HYVW1AD-NL9v6blu0jF2dElT7idV_5VbLsXD6vpFmW6Knzr3d7iy7KsytuU6mHXT11v_m; QN267=085053407291d013c7; QN58=1685759689485%7C1685763244022%7C12; JSESSIONID=22D7903E31288266A0301A5F89AADD87; __qt=v1%7CVTJGc2RHVmtYMStoSGxOT3A4RzZGN05DNllpRzlqMm1kM0x5VXAxdGFJdUh4SXZJZVZZOUFrdjA1cmhSeFUwd3lVbDA4UWVrb3J4ZDhZbytYcDdMM1NWT2hpNXJ0SmYwb1dnb3lNMjNhWXY1R0pKM21kUlE5MW1GOGorZ1RMbjI5OTgzZ0NQaVlPZVdHOXdQRmxnN1FtRXZFQWFBLzByZjRKNXNpclppV1ljPQ%3D%3D%7C1685763244171%7CVTJGc2RHVmtYMS9IM2FHam5MNUpEVFEySngvV3NZYWtQaTlUNGRMNThSQ3VQRUo5cGFSTkorTUlhWGVvUUlzQUQwNmJTK21haFh4M01tYXRDSFZ1Umc9PQ%3D%3D%7CVTJGc2RHVmtYMSsvNjh2VTBDVmNTY3F0WE9CTWdPUWtCcFFNcDFLdjJrSXB0cUhLSGVJR09lbXBiaVczRlhTcGQyKzRXbkpTek1XTC9oRlZoM2xYMng4SVVMRGZlaHJrU3VtRkdUdUx5WGpDN0U2NWVERWVKenkrOTNGUURaSy9EeWVpcnQrUXlZVk5mY3NiaDFpdzVSTnRmWTRwUEkxN2lmdlI5cW5KcEp5Q2RiT1YxQ1BSSEM2MFZSZDNuLy96T3kwYklIcm9uOWpqZnI3bTBMWW93ZWZyMmEyQ21HUW5NbHJWYVR5Q3VIMXNjSzNTTXFYWUpLTG5HdVh0YnBueUk0VDlHSWpENWxyTnRFays0NytlS1ZyQmxrTmdwbEo5OFhGbktWOEFCK2IvbEpiZG4rTzFxWlgrQ2pmMUl6ZkdMeEErNUVyVjBEaHhWN1AzMC9Nd0ZQK2xWelNUVUgyNW0rTUVlQTVKbGlhR0tST29oRzRhempOZVVESG1LMDlTdWlWQXpHWnErQXk3YmdCQWQwZG81dkhoc1NsQW9YaVJBOWlQZjN5MEpsRzJtS2gyeXY2QzE0OHArbDFoNGlVcW5MWGFkM080SjErck56bnZsTi9FcjNZOWtSMUNHYS9Dd0QrQ09LY3RpNmZCSE9GUWFwV3NaUzZQRGZOakxLMHhRTmo1ZUNhSko3ZDZEWXdHeWdpYURDUVpmenJtK1hwUjFLdU04UFFBdUNKVlR5VlRaYlpsMU9BZmpqd3lkT2paQzJlQUVjM0J2cG1FKzlYcXlQUVVjbVc3UkJ1b1U3RU1BVHVQemJzR0FiU2hCZXNWam9FTjJqa2IrSVQ0Q3h4SGVleHlOVDRYT2R3K2hOYzc1TEZIQ3lPSWRLM1NNcG5mUWJkRVNiNTZxZ0ZwZnZtZEhzNFBKWk5hWHJ6MDI4aGhuYWhrWDFjdDQ4S3Z0aEpqMjVSenZVaVdQMjJ3VzBvNUhoenpNTVVhU0dvPQ%3D%3D; Hm_lpvt_15577700f8ecddb1a927813c81166ade=1685763244; QN271=83df3432-fa03-46bb-a8e5-df1a3312d171',
        }
        response = requests.get(url=url, headers=headers)
        selector = parsel.Selector(response.text)
        try:
            lis = selector.css('.b_strategy_list .list_item .tit')
        except:
            continue
        # 模式a代表末行追加
        with open('./data/traj.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['user', 'day', 'seq', 'name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            for li in lis:
                link = li.get()
                index1 = link.find('gonglve')
                index2 = link.find('">', index1)
                sub_url = 'https://travel.qunar.com/travelbook/' + link[index1+8:index2]
                sub_headers = {
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
                    'Cookie': 'SECKEY_ABVK=xjHYNyvTER514p26YGWC343OQQWweJehu66r/6cPc24%3D; BMAP_SECKEY=y0qt7sYdJfhqzvqZ18l67KmoJlBuqetWD1zdz5MzwuZrQHvAwF6UUZ9zuKNRvz9pQEvygHkfFMg7RTqqxusNqVCSbZh9q_jHzbKvreaWNS2Ki_fgaM9PUa-LxFRWBpYjj2b_AS5EMFSJvegKEMYmSOA5ll-No9gP4F-vAHnGvhb5l17PnvU8aVvLT2o30cj-; QN1=00008c002eb4517eb5a80697; QN99=6026; QunarGlobal=10.66.70.17_7b56d759_18879a3b151_-210e|1685672123197; qunar-assist={%22version%22:%2220211215173359.925%22%2C%22show%22:false%2C%22audio%22:false%2C%22speed%22:%22middle%22%2C%22zomm%22:1%2C%22cursor%22:false%2C%22pointer%22:false%2C%22bigtext%22:false%2C%22overead%22:false%2C%22readscreen%22:false%2C%22theme%22:%22default%22}; csrfToken=5jYhreufGA78TXFn4YaTat5MVYZwDgya; QN601=5cfdfde462b48b806337ef3c5d223618; _i=VInJOmiPqzxq2ZR1YnO78qZyXWtq; QN163=0; QN269=55D9685000EB11EE9B7CFA163E8F381D; fid=c24aeb7c-c5a0-486d-8072-0817e281a724; QN48=tc_9c1923fcdb9ffb69_18879e3687e_3940; ariaDefaultTheme=null; QN243=44; QN57=16856739703210.8698434013169207; Hm_lvt_15577700f8ecddb1a927813c81166ade=1685673970; QN63=%E6%B2%B3%E6%BA%90; QN300=s%3Dbing; QN71=MTEzLjgxLjMuMjEyOuaDoOW3njox; QN205=s%3Dbing; QN277=s%3Dbing; _vi=mWJI1m2_O13zAn2BVaJgsh9oiy2HENM6-UJhhjMeQBCgf0pG6Cr0bvzKWAB1JpR-DS_3uI0Y31Yt4bRAjNKFdDRAaRIDfLq_1SB027_HYVW1AD-NL9v6blu0jF2dElT7idV_5VbLsXD6vpFmW6Knzr3d7iy7KsytuU6mHXT11v_m; QN267=085053407291d013c7; QN58=1685759689485%7C1685763244022%7C12; JSESSIONID=22D7903E31288266A0301A5F89AADD87; __qt=v1%7CVTJGc2RHVmtYMStoSGxOT3A4RzZGN05DNllpRzlqMm1kM0x5VXAxdGFJdUh4SXZJZVZZOUFrdjA1cmhSeFUwd3lVbDA4UWVrb3J4ZDhZbytYcDdMM1NWT2hpNXJ0SmYwb1dnb3lNMjNhWXY1R0pKM21kUlE5MW1GOGorZ1RMbjI5OTgzZ0NQaVlPZVdHOXdQRmxnN1FtRXZFQWFBLzByZjRKNXNpclppV1ljPQ%3D%3D%7C1685763244171%7CVTJGc2RHVmtYMS9IM2FHam5MNUpEVFEySngvV3NZYWtQaTlUNGRMNThSQ3VQRUo5cGFSTkorTUlhWGVvUUlzQUQwNmJTK21haFh4M01tYXRDSFZ1Umc9PQ%3D%3D%7CVTJGc2RHVmtYMSsvNjh2VTBDVmNTY3F0WE9CTWdPUWtCcFFNcDFLdjJrSXB0cUhLSGVJR09lbXBiaVczRlhTcGQyKzRXbkpTek1XTC9oRlZoM2xYMng4SVVMRGZlaHJrU3VtRkdUdUx5WGpDN0U2NWVERWVKenkrOTNGUURaSy9EeWVpcnQrUXlZVk5mY3NiaDFpdzVSTnRmWTRwUEkxN2lmdlI5cW5KcEp5Q2RiT1YxQ1BSSEM2MFZSZDNuLy96T3kwYklIcm9uOWpqZnI3bTBMWW93ZWZyMmEyQ21HUW5NbHJWYVR5Q3VIMXNjSzNTTXFYWUpLTG5HdVh0YnBueUk0VDlHSWpENWxyTnRFays0NytlS1ZyQmxrTmdwbEo5OFhGbktWOEFCK2IvbEpiZG4rTzFxWlgrQ2pmMUl6ZkdMeEErNUVyVjBEaHhWN1AzMC9Nd0ZQK2xWelNUVUgyNW0rTUVlQTVKbGlhR0tST29oRzRhempOZVVESG1LMDlTdWlWQXpHWnErQXk3YmdCQWQwZG81dkhoc1NsQW9YaVJBOWlQZjN5MEpsRzJtS2gyeXY2QzE0OHArbDFoNGlVcW5MWGFkM080SjErck56bnZsTi9FcjNZOWtSMUNHYS9Dd0QrQ09LY3RpNmZCSE9GUWFwV3NaUzZQRGZOakxLMHhRTmo1ZUNhSko3ZDZEWXdHeWdpYURDUVpmenJtK1hwUjFLdU04UFFBdUNKVlR5VlRaYlpsMU9BZmpqd3lkT2paQzJlQUVjM0J2cG1FKzlYcXlQUVVjbVc3UkJ1b1U3RU1BVHVQemJzR0FiU2hCZXNWam9FTjJqa2IrSVQ0Q3h4SGVleHlOVDRYT2R3K2hOYzc1TEZIQ3lPSWRLM1NNcG5mUWJkRVNiNTZxZ0ZwZnZtZEhzNFBKWk5hWHJ6MDI4aGhuYWhrWDFjdDQ4S3Z0aEpqMjVSenZVaVdQMjJ3VzBvNUhoenpNTVVhU0dvPQ%3D%3D; Hm_lpvt_15577700f8ecddb1a927813c81166ade=1685763244; QN271=83df3432-fa03-46bb-a8e5-df1a3312d171',
                }
                sub_response = requests.get(url=sub_url, headers=sub_headers)
                sub_selector = parsel.Selector(sub_response.text)
                items = sub_selector.css('.list.js_expand_item')
                if len(items.getall()) != 0:
                    user += 1
                day = 0
                for item in items:
                    day += 1
                    names = item.css('.des.clrfix a::text').getall()
                    seq = 0
                    for name in names:
                        seq += 1
                        dit = {
                            'user': user,
                            'day': day,
                            'seq': seq,
                            'name': name,
                        }
                        writer.writerow(dit)


# 利用高德API，根据名称获取经纬度
def get_xy():
    filename = './data/test.csv'
    places = []
    with open(filename) as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            places.append(row[1])
    places = places[1:]     # 要查询的地点名称
    url = 'https://restapi.amap.com/v3/geocode/geo'
    locations = []          # 经纬度
    for place in places:
        params = {'key': '5d5ca7f300800fb6dbefd46be33986ea',
                  'address': place,
                  'city': '厦门'}
        res = requests.get(url, params)
        jd = json.loads(res.text)
        # 查询到了状态为1，未查询到状态为0
        if jd['status'] == '1':
            locations.append(jd['geocodes'][0]['location'])
        else:
            locations.append(0)
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for loc in locations:
            writer.writerow([loc])


if __name__ == '__main__':
    # get_recommend_time()
    # get_traj()
    get_xy()

