#include <iostream>
#include <algorithm>

using namespace std;

/************************************************************************/
/* ��������                                                               */
/************************************************************************/
const int N = 50;//Ϊ���㷨д�����򵥣����ﶨ��һ���㹻����������洢����(Ϊ�˱�����������ж�̬����ռ䣬�����Ļ��㷨�������Ƚ��鷳������ֻ��Ϊ���㷨��������)
int data[N][N];//�洢����ԭʼ����
int dp[N][N];//�洢��̬�滮�����е�����
int n;//���Ĳ���

/*��̬�滮ʵ���������*/
void tower_walk()
{
    // dp��ʼ��
    for (int i = 0; i < n; ++i)
    {
        dp[n - 1][i] = data[n - 1][i];
    }
    int temp_max;
    for (int i = n - 1; i >= 0; --i)
    {
        for (int j = 0; j <= i; ++j)
        {
            // ʹ�õ��ƹ�ʽ����dp��ֵ
            temp_max = max(dp[i + 1][j], dp[i + 1][j + 1]);
            dp[i][j] = temp_max + data[i][j];
        }
    }
}

/*��ӡ���ս��*/
void print_result()
{
    cout << "���·���ͣ�" << dp[0][0] << '\n';
    int node_value;
    // �����������Ԫ��
    cout << "���·����" << data[0][0];
    int j = 0;
    for (int i = 1; i < n; ++i)
    {
        node_value = dp[i - 1][j] - data[i - 1][j];
        /* ���node_value == dp[i][j]��˵����һ��Ӧ����data[i][j]�����node_value == dp[i][j + 1]��˵����һ��Ӧ����data[i][j + 1]*/
        if (node_value == dp[i][j + 1]) ++j;
        cout << "->" << data[i][j];
    }
    cout << endl;
}

int main()
{
    cout << "�������Ĳ�����";
    cin >> n;
    cout << "�������Ľڵ�����(��i����i���ڵ�)��\n";
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            cin >> data[i][j];
        }
    }

    tower_walk();
    print_result();
}
