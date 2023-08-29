//德科2 lins
#include<bits/stdc++.h>
using namespace std;
int find_same(int a, int b, int c, int d, int e, int f) {
    int count_long=0;
    int n[3][1000];
    for (int j = a; j <= b; j++) n[0][j] = 1;
    for (int j = c; j <= d; j++) n[1][j] = 1;
    for (int j = e; j <= f; j++) n[2][j] = 1;

    for (int j = 0; j < 1000; j++) {
        int flag = 0;
        for (int i = 0; i < 3; i++)
        {
            if (n[i][j] != 1)flag = 1;
        }
        if (flag == 0)count_long++;
    }

    return count_long;

}
int main() {
    int a[3][4];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            cin >> a[i][j];
        }
    }
    int y, x;
    y = find_same(a[0][1] - a[0][3], a[0][1], a[1][1] - a[1][3], a[1][1], a[2][1] - a[2][3], a[2][1]);
    x = find_same(a[0][0], a[0][0] + a[0][2], a[1][0], a[1][0] + a[1][2], a[2][0], a[2][0] + a[2][2]);
    cout << x * y;
    return 0;
}


//德科2

#include<bits/stdc++.h>
using namespace std;
struct node {
    int li;
    int lj;
    int ri;
    int rj;
};
node nums[3];
int minLi = INT_MAX, maxLj = INT_MIN, maxRi = INT_MIN, minRj = INT_MAX;

bool isRight(int pi, int pj) {
    int flag = 0;
    for (int i = 0; i < 3; i++) {
        if (pi >= nums[i].li && pi + 1 <= nums[i].ri && pj >= nums[i].rj && pj + 1 <= nums[i].lj) {
            flag++;
        }
    }
    if (flag == 3) {
        //         cout<<pi<<pj;
        return true;
    }
    return false;
}

int main() {
    for (int i = 0; i < 3; i++) {
        cin >> nums[i].li;
        cin >> nums[i].lj;
        int h, w;
        cin >> w;
        cin >> h;
        nums[i].ri = nums[i].li + w;
        nums[i].rj = nums[i].lj - h;

        minLi = min(minLi, nums[i].li);
        maxLj = max(maxLj, nums[i].lj);
        maxRi = max(maxRi, nums[i].ri);
        minRj = min(minRj, nums[i].rj);
    }
    int count = 0;
    for (int i = minLi; i <= maxRi; i++) {
        for (int j = minRj; j <= maxLj; j++) {
            if (isRight(i, j)) {
                count++;
            }
        }
    }
    //     isRight(3, 2);
    //     cout<<minLi<<maxLj<<maxRi<<minRj<<endl;
    cout << count;
}


//德科1

#include<bits/stdc++.h>
using namespace std;
int main() {
    int k, N;
    cin >> k;
    cin >> N;
    vector<string> list;
    for (int i = 0; i < N; i++) {
        string temp;
        cin >> temp;
        list.push_back(temp);
    }
    string res;
    string head = list[k];
    res += head;
    list.erase(list.begin() + k);
    char tail = head[head.size() - 1];
    while (true) {
        set<string> mset;

        for (int i = 0; i < list.size(); i++) {
            string word = list[i];
            if (word.find(tail) == 0) {
                mset.insert(word);
            }
        }
        if (mset.size() == 0)break;
        string first = *(mset.begin());
        int len = first.size();
        string aim = "";
        for (string s : mset) {
            if (s.length() > len) {
                len = s.size();
                aim = s;
            }
        }

        string into = len != first.size() ? aim : first;
        tail = into[into.size() - 1];
        res += into;

        for (int i = 0; i < list.size(); i++) {
            if (list[i] == into) {
                list.erase(list.begin() + i);
                i--;
            }
        }
    }

    cout << res;

}