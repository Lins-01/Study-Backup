#include<stdio.h>

int able(int num[]);

int main()
{
	int num[10];
	int i=0;
	//��1~10ȫ��������浽������ �����ö������� ��������� 
	for(int a=1;a<=10;a++)
	{
		i=0;
		num[i]=a;
		for(int b=1;b<=10;b++)
		{
			if(b!=a)
			{
				i=1;
				num[i]=b;
				for(int c=1;c<=10;c++)
				{
					if(c!=a&&c!=b)
					{
						i=2;
						num[i]=c;
						for(int d=1;d<=10;d++)
						{
							if(d!=a&&d!=b&&d!=c)
							{
								i=3;
								num[i]=d;
								for(int e=1;e<=10;e++)
								{
									if(e!=a&&e!=b&&e!=c&&e!=d)
									{
										i=4;
										num[i]=e;
										for(int f=1;f<=10;f++)
										{
											if(f!=a&&f!=b&&f!=c&&f!=d&&f!=e)
											{
												i=5;
												num[i]=f;
												for(int g=1;g<=10;g++)
												{
													if(g!=a&&g!=b&&g!=c&&g!=d&&g!=e&&g!=f)
													{
														i=6;
														num[i]=g;
														for(int h=1;h<=10;h++)
														{
															if(h!=a&&h!=b&&h!=c&&h!=d&&h!=e&&h!=f&&h!=g)
															{
																i=7;
																num[i]=h;
																for(int x=1;x<=10;x++)
																{
																	if(x!=a&&x!=b&&x!=c&&x!=d&&x!=e&&x!=f&&x!=g&&x!=h)
																	{
																		i=8;
																		num[i]=x;
																		for(int y=1;y<=10;y++)
																		{
																			if(y!=a&&y!=b&&y!=c&&y!=d&&y!=e&&y!=f&&y!=g&&y!=h&&y!=x)
																			{
																				i=9;
																				num[i]=y;
																				//�ж�ÿ�������Ƿ�����Ҫ��
																				if(able(num)) //���������
																				{
																					for(int j=0;j<10;j++)
																					{	
																						//���������ʽ ���һ�����к��� 
																						if(j==9)
																						{
																							printf("%d\n",num[j]);
																							break;
																						}
																						printf("%d ",num[j]);
																						
																					}
																				} 
																			}
																		}
																	}
																}
															}
														}
													}
												}
											} 
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	 
	return 0;
} 

int able(int num[])
{
	
	
	//���ҵ�����������6��λ��  �ٶ�ÿ��Ҫ�������ж� �����㷵��0 
	for(int i=0;i<10;i++)
	{
		if(num[i]==6)
		{
			//6���5 
			if((i+1==10)||num[i+1]!=5)
			{
				return 0;//able�������� 
			}
			//1,2,3����Ϊ6 
			if((num[i-1]==1)||(num[i-1]==2)||(num[i-1]==3))
			{
				return 0;//able��������
			}
			//5����Ϊ7,8  ��ִ�е������˵���Ѿ������������������� 
			if((num[i+2]==7)||(num[i+2]==8))
			{
				return 0;//able��������
			}
		}
	}
	return 1;//ִ�е�����˵��������������Ҫ��  ������� 
}





















