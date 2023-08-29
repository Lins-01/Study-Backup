#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>
using namespace std;
string X,Y;
int dp[10000][10000];
int main(){
	cin>>X>>Y;
	int len1 = X.length(),len2 = Y.length();
	for(int i = 1; i <= len1; i++){
		dp[i][0] = 0;
		dp[0][i] = 0;
	}
	for(int i = 1; i <= len1; i++){
		for(int j = 1; j <= len2; j++){
			if(X[i-1] == Y[j-1])
					dp[i][j] = dp[i-1][j-1]+1;
			else
				dp[i][j] = max(dp[i][j-1],dp[i-1][j]); 
			
		}
	}

	cout<<dp[len1][len2];
	return 0;
} 
