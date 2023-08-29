import requests

page_limit =50
page_start =0

all_hot_movies = []

while True:
    print(f'get page:{page_start}')
    # 这里面信息都是从f12-》network-》Fetch/XHR里面找的 
    # headers每个请求里面都有的
    resp = requests.get(
        f'https://movie.douban.com/j/search_subjects?type=movie&tag=%E7%83%AD%E9%97%A8&page_limit={page_limit}&page_start={page_start}',
        headers={
            'User-Agent':
'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
        }
    )

    if resp.status_code !=200 or len(resp.json()['subjects'])==0:
        break

    page_start += page_limit
    all_hot_movies += resp.json()['subjects']




print(len(all_hot_movies))

print(all_hot_movies )