#include<iostream>
#include<cstdlib>
#define MAXV 100
#define InfoType char
#define Vertex char
using namespace std; 
int visited[MAXV];//锟斤拷锟矫凤拷锟绞憋拷志锟斤拷锟介，为锟斤拷锟斤拷同一锟斤拷锟姐被锟截革拷锟斤拷锟斤拷
int visited2[MAXV];
// 图锟斤拷锟节接撅拷锟斤拷锟斤拷锟酵讹拷锟斤拷 
typedef struct
{
	int no;//锟斤拷锟斤拷锟斤拷 
	//InfoType info;//锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷息 权锟斤拷 
} VertexType;//锟斤拷锟斤拷锟斤拷锟斤拷
typedef struct
{
	int edges[MAXV][MAXV];//锟节接撅拷锟斤拷谋锟斤拷锟斤拷锟�
	int n,e;//锟斤拷锟斤拷锟斤拷 锟斤拷锟斤拷 
	VertexType vexs[MAXV];// 锟斤拷哦锟斤拷锟斤拷锟较� 
} MGraph;//锟斤拷锟斤拷锟斤拷图锟节接撅拷锟斤拷锟斤拷锟斤拷 

//图锟斤拷锟节接憋拷锟斤拷锟酵讹拷锟斤拷                                                
typedef struct ANode
{
	int adjvex;//锟矫边碉拷锟秸碉拷锟斤拷
	struct ANode *nextarc;//指锟斤拷锟斤拷一锟斤拷锟竭碉拷指锟斤拷
	//InfoType info;//锟矫边碉拷锟斤拷锟斤拷锟较� 权锟斤拷 
	 
} ArcNode;//锟竭节碉拷锟斤拷锟斤拷
typedef struct Vnode
{
	char data;//锟斤拷锟斤拷锟斤拷息
	ArcNode * firstarc; //指锟斤拷锟揭伙拷锟斤拷锟� 
} VNode;//锟节接憋拷头锟节碉拷锟斤拷锟斤拷  
//typedef VNode AdjList[MAXV];//AdjList锟斤拷锟节接憋拷锟斤拷锟斤拷
typedef struct
{
	VNode adjlist[MAXV];//锟节接憋拷
	int n,e;//图锟叫讹拷锟斤拷锟斤拷n锟酵憋拷锟斤拷e 
} ALGraph;//锟斤拷锟斤拷锟斤拷图锟节接憋拷锟斤拷锟斤拷 

//锟节接撅拷锟斤拷转锟节接憋拷
void MatToList(MGraph g,ALGraph *&G);



void DFS(ALGraph *G,int v);//v锟斤拷始锟斤拷锟斤拷锟斤拷

void BFS (ALGraph *G,int v);

//模锟斤拷锟斤拷锟� 锟斤拷锟斤拷P208 A1锟斤拷锟斤拷 
int M[5][5]={
{0,1,0,1,1},
{1,0,1,1,0},
{0,1,0,1,1},
{1,1,1,0,1},
{1,0,1,1,0},
} ;

int main()
{
	//图锟侥达拷锟斤拷
	//指锟斤拷要锟斤拷锟斤拷始值为NULL 
	MGraph g;//锟斤拷锟斤拷锟节接撅拷锟斤拷
	ALGraph *G=NULL;//锟斤拷锟斤拷锟节接憋拷
	
	
	int n,e;
	//锟斤拷始锟斤拷锟节接撅拷锟斤拷
	//转锟斤拷为锟节接憋拷
	cout<<"���붥����n:";//锟斤拷锟斤拷n=5 
	cin>>n;
	cout<<"�������e:";//锟斤拷锟斤拷e=16
	cin>>e;
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			{
				//cout<<"锟斤拷锟斤拷"<<(i+1)<<"锟诫顶锟斤拷"<<(j+1)<<"之锟斤拷叩锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷0为锟睫ｏ拷1为锟斤拷:";
				//cin>>g.edges[i][j];
				//锟斤拷锟缴简单碉拷锟斤拷锟斤拷
				 g.edges[i][j]=M[i][j];
			}
	g.n=n;g.e=e;
	MatToList(g,G);

	//图锟侥憋拷锟斤拷
	//DFS
	cout<<"DFS:"; 
	DFS(G,2);//锟接讹拷锟斤拷0锟斤拷始锟斤拷锟斤拷 //锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟接改成从讹拷锟斤拷2锟斤拷始锟斤拷锟斤拷 
	cout<<endl; 
	cout<<"BFS:"; 
	BFS(G,2);
	return 0;
}

void MatToList(MGraph g,ALGraph *&G)
{
	int i,j;
	ArcNode *p;
	G=(ALGraph *)malloc(sizeof(ALGraph));
	for(i=0;i<g.n;i++)//锟斤拷锟节接憋拷锟斤拷锟斤拷锟斤拷头锟节碉拷锟街革拷锟斤拷锟斤拷贸锟街�
	{
		G->adjlist[i].firstarc=NULL;
	}
	//锟斤拷锟斤拷锟斤拷每一锟斤拷 锟斤拷应锟节接憋拷锟斤拷每一锟斤拷锟斤拷锟姐建锟斤拷锟侥碉拷锟斤拷锟斤拷
	for(i=0;i<g.n;i++)//锟斤拷锟斤拷诮泳锟斤拷锟斤拷锟矫匡拷锟皆拷锟�
	{
		for(j=g.n-1;j>=0;j--)
		{
			if(g.edges[i][j]!=0)//锟斤拷锟斤拷一锟斤拷锟斤拷
			{
				p=(ArcNode *)malloc(sizeof(ArcNode));//锟斤拷锟斤拷一锟斤拷锟节碉拷*p
				p->adjvex=j;//锟斤拷锟斤拷帽叩锟斤拷盏锟斤拷锟� 锟斤拷j
				p->nextarc=G->adjlist[i].firstarc;//锟斤拷锟斤拷头锟藉法锟斤拷锟斤拷*p
				G->adjlist[i].firstarc=p;
			}
		}
	}
	G->n=g.n;
	G->e=g.e;
}


void DFS(ALGraph *G,int v)
{
	ArcNode *p;
	visited[v]=1;//锟斤拷锟窖凤拷锟绞憋拷锟�
	cout<<v;//锟斤拷锟斤拷锟斤拷锟斤拷识锟斤拷锟侥憋拷锟�
	p=G->adjlist[v].firstarc;//p指锟津顶碉拷v锟侥碉拷一锟斤拷锟节接碉拷
	while (p!=NULL)
	{
		if(visited[p->adjvex]==0)
				DFS(G,p->adjvex);//锟斤拷p->adjvex锟斤拷锟斤拷未锟斤拷锟绞ｏ拷锟捷癸拷锟斤拷锟斤拷锟�
		p=p->nextarc;//p指锟津顶碉拷v锟斤拷锟斤拷一锟斤拷锟节接碉拷
	}
	
}

 

void BFS (ALGraph *G,int v)
{
	ArcNode *p;
	int queue[MAXV],front=0,rear=0;//定义循环队列并初始化队头队尾
	int w;
	cout<<v;//输出被访问顶点的编号
	visited2[v]=1;//置已访问标记
	rear=(rear+1)%MAXV;
	queue[rear]=v;//v����
	while (front!=rear)//�����в�Ϊ��ʱѭ��
	{
		front=(front+1)%MAXV;
		w=queue[front];
		p=G->adjlist[w].firstarc;
		while (p!=NULL)
		{
			if(visited2[p->adjvex]==0)
			{
				cout<<p->adjvex;
				visited2[p->adjvex]=1;
				rear=(rear+1)%MAXV;
				queue[rear]=p->adjvex;
			}
			p=p->nextarc;
		}
		
	}
	cout<<endl;
}




















