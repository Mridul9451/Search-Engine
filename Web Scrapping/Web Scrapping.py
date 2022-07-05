import time
from selenium import webdriver
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import json

driver = webdriver.Chrome(ChromeDriverManager().install())

cnt1 = 0

for i in range(1, 20):
    cnt = 0
    driver.get("https://leetcode.com/problemset/all/?page="+str(i))
    time.sleep(5)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    all_ques_div = soup.findAll("div", {"class": "truncate"})

    all_ques = []
# print(all_ques_div)

    for ques in all_ques_div:
        all_ques.append(ques.find("a"))

# print(all_ques[0])

    urls = []
    titles = []
    problem_description = []

    for ques in all_ques:
        urls.append("https://leetcode.com"+ques['href'])

    for ques in all_ques:
        flag = 0
        correct = ""
        splits = ques.text.split()
        for split in splits:
            if flag == 0:
                flag = 1
            else:
                correct = correct + split + " "
        titles.append(correct)

    with open("problem_urls.txt", "a+") as f:
        f.write('\n'.join(urls))
        f.write('\n')

    with open("problem_titles.txt", "a+") as f:
        f.write('\n'.join(titles))
        f.write('\n')

    for url in urls:
        driver.get(url)
        cnt += 1
        cnt1 += 1
        time.sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        if soup.find('div', {"class": "description__24sA"}):
            problem_text = soup.find('div', {"class": "content__u3I1"}).get_text()
            # print(problem_text)
            problem_text = problem_text.encode("utf-8")
            problem_text = str(problem_text)
        else:
            problem_text = ""

        problem_description.append(problem_text)

        with open("problem" + str(cnt1) + ".txt", "w+") as f:
            f.write(problem_text)

    listObj = []

    print(cnt1)

    for j in range(cnt):
        # print(urls[j])
        dictionary = {
            "URL": urls[j],
            "Title": titles[j],
            "Description": problem_description[j]
        }
        listObj.append(dictionary)

    with open("sample.json", 'a+') as json_file:
        json.dump(listObj, json_file, indent=4, separators=(',', ': '))