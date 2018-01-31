function [result]=preded(W, weght, test1,test2,ptop)

alp=weght.alp;

dataA_1=test1.dataA_1;
dataA_2=test1.dataA_2;
dataB_1=test2.dataB_1;
dataB_2=test2.dataB_2;
dataA_3=test1.dataA_3;
dataB_3=test2.dataB_3;
dataA_4=test1.dataA_4;
dataB_4=test2.dataB_4;
dataA_5=test1.dataA_5;
dataB_5=test2.dataB_5;

num= size(dataA_1,1);

for i=1:num
    
    testx1= repmat(dataA_1(i,:),num,1);
    testx2= repmat(dataA_2(i,:),num,1);
    testx3= repmat(dataA_3(i,:),num,1);
    testx4= repmat(dataA_4(i,:),num,1);
    testx5= repmat(dataA_5(i,:),num,1);
    
    dist_gd= alp(1)*sum(abs(testx1*W.WA_1-dataB_1*W.WB_1).^2,2).^(1/2)+alp(2)*sum(abs(testx2*W.WA_2-dataB_2*W.WB_2).^2,2).^(1/2)...
        +alp(3)*sum(abs(testx3*W.WA_3-dataB_3*W.WB_3).^2,2).^(1/2)+alp(4)*sum(abs(testx4*W.WA_4-dataB_4*W.WB_4).^2,2).^(1/2)...
        +alp(5)*sum(abs(testx5*W.WA_5-dataB_5*W.WB_5).^2,2).^(1/2);
    
    dist_LUS= sum(abs(testx1*W.WA_1-dataB_1*W.WB_1).^2,2).^(1/2)+sum(abs(testx2*W.WA_2-dataB_2*W.WB_2).^2,2).^(1/2)...
        +sum(abs(testx3*W.WA_3-dataB_3*W.WB_3).^2,2).^(1/2)+sum(abs(testx4*W.WA_4-dataB_4*W.WB_4).^2,2).^(1/2)...
        +sum(abs(testx5*W.WA_5-dataB_5*W.WB_5).^2,2).^(1/2);
    
    
    dist_noLUS=sum(abs([testx1 testx2 testx3 testx4 testx5]-[dataB_1 dataB_2 dataB_3 dataB_4 dataB_5]).^2,2).^(1/2);
    
    [rank_result_gd,I_gd(i,:)] = sort(dist_gd,'ascend');
    [rank_result_LUS,I_LUS(i,:)] = sort(dist_LUS,'ascend');
    [rank_result_noLUS,I_noLUS(i,:)] = sort(dist_noLUS);
    
    rankindex_gd(i)=  find(I_gd(i,:)==i);
    rankindex_LUS(i)=  find(I_LUS(i,:)==i);
    rankindex_noLUS(i)=  find(I_noLUS(i,:)==i);
    
end
pre_gd=0;
pre_LUS=0;
pre_noLUS=0;

mRR_gd=0;
mRR_LUS=0;
mRR_noLUS=0;


indexno=find(rankindex_gd<=ptop);
result.pre_gd=length(indexno)/num;

indexma=find(rankindex_LUS<=ptop);
result.pre_LUS=length(indexma)/num;


indexno=find(rankindex_noLUS<=ptop);
result.pre_noLUS=length(indexno)/num;

result.mRR_dg=sum(rankindex_gd.^-1)/num;
result.mRR_LUS=sum(rankindex_LUS.^-1)/num;
result.mRR_noLUS=sum(rankindex_noLUS.^-1)/num;




