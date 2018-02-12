index=randperm(size(facebook_screenname,1));


star_num=1;
end_num=3000;

dataA_1=facebook_username(index(star_num:end_num),:);
dataB_1=twitter_username(index(star_num:end_num),:);

dataA_2=facebook_screenname(index(star_num:end_num),:);
dataB_2=twitter_screenname(index(star_num:end_num),:);

dataA_3=facebook_content(index(star_num:end_num),:);
dataB_3=twitter_content(index(star_num:end_num),:);

dataA_4=facebook_net(index(star_num:end_num),:);
dataB_4=twitter_net(index(star_num:end_num),:);

dataA_5=facebook_description(index(star_num:end_num),:);
dataB_5=twitter_description(index(star_num:end_num),:);


%%%%%%%%%%%training data%%%%%%%%%%%%%%%%%%%%%%%%%

dataA_1do=[];
dataB_1do=[];
dataBwrong_1do=[];

dataA_2do=[];
dataB_2do=[];
dataBwrong_2do=[];

dataA_3do=[];
dataB_3do=[];
dataBwrong_3do=[];

dataA_4do=[];
dataB_4do=[];
dataBwrong_4do=[];

dataA_5do=[];
dataB_5do=[];
dataBwrong_5do=[];


doo=10; %%%  nonmatching pairs

for i=1:size(dataA_1,1)

   indexmiss=randperm(size(dataA_1,1));

   dataA_1do= [dataA_1do;repmat(dataA_1(i,:),doo,1)];
   dataB_1do = [dataB_1do;repmat(dataB_1(i,:),doo,1)];
   dataBwrong_1do=[dataBwrong_1do; dataB_1(indexmiss(1:doo),:)];

   dataA_2do= [dataA_2do;repmat(dataA_2(i,:),doo,1)];
   dataB_2do = [dataB_2do;repmat(dataB_2(i,:),doo,1)];
   dataBwrong_2do=[dataBwrong_2do; dataB_2(indexmiss(1:doo),:)];

   dataA_3do= [dataA_3do;repmat(dataA_3(i,:),doo,1)];
   dataB_3do = [dataB_3do;repmat(dataB_3(i,:),doo,1)];
   dataBwrong_3do=[dataBwrong_3do; dataB_3(indexmiss(1:doo),:)];

   dataA_4do= [dataA_4do;repmat(dataA_4(i,:),doo,1)];
   dataB_4do = [dataB_4do;repmat(dataB_4(i,:),doo,1)];
   dataBwrong_4do=[dataBwrong_4do; dataB_4(indexmiss(1:doo),:)];

   dataA_5do= [dataA_5do;repmat(dataA_5(i,:),doo,1)];
   dataB_5do = [dataB_5do;repmat(dataB_5(i,:),doo,1)];
   dataBwrong_5do=[dataBwrong_5do; dataB_5(indexmiss(1:doo),:)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%testing data%%%%%%%%%%%%%%%%%%%%%%%%%
star_num=3001;
end_num=3600;

test1.dataA_1=facebook_username(index(star_num:end_num),:);
test1.dataA_2=facebook_screenname(index(star_num:end_num),:);
test1.dataA_3=facebook_content(index(star_num:end_num),:);
test1.dataA_4=facebook_net(index(star_num:end_num),:);
test1.dataA_5=facebook_description(index(star_num:end_num),:);


test2.dataB_1=twitter_username(index(star_num:end_num),:);
test2.dataB_2=twitter_screenname(index(star_num:end_num),:);
test2.dataB_3=twitter_content(index(star_num:end_num),:);
test2.dataB_4=twitter_net(index(star_num:end_num),:);
test2.dataB_5=twitter_description(index(star_num:end_num),:);

[n1,m1]=size(dataA_1);
[n2,m2]=size(dataA_2);
[n3,m3]=size(dataA_3);
[n4,m4]=size(dataA_4);
[n5,m5]=size(dataA_5);

d=300;

WA_1=eye(m1,d);
WA_1_t0=eye(m1,d);
WB_1=eye(m1,d);
WB_1_t0=eye(m1,d);

WA_2=eye(m2,d);
WB_2=eye(m2,d);
WB_2_t0=WB_2;
WA_2_t0=WA_2;

WA_3=eye(m3,d);
WA_3_t0=WA_3;
WB_3=eye(m3,d);
WB_3_t0=WB_3;

WB_4=eye(m4,d);
WA_4=eye(m4,d);
WB_4_t0=WB_4;
WA_4_t0=WA_4;

WB_5=eye(m5,d);
WA_5=eye(m5,d);
WB_5_t0=WB_5;
WA_5_t0=WA_5;

W.WA_1=WA_1;
W.WA_2=WA_2;
W.WA_3=WA_3;
W.WA_4=WA_4;
W.WA_5=WA_5;

W.WB_1=WB_1;
W.WB_2=WB_2;
W.WB_3=WB_3;
W.WB_4=WB_4;
W.WB_5=WB_5;

%%%%%%%parametres%%%%
topk=10;
weight.alp=[0.2 0.2 0.2 0.2 0.2];
leaning_rate1=0.00001;
leaning_rate2=0.00001;
iteration_k1=3;
iteration_k2=3;
C=1000;
c=1;
%%%%%%
% [result]=preded(W, weight, test1,test2,topk)
weightchange(1,:)=weight.alp;
%%%%%%%%%%%%%%%%%%%%%%%%%Tarining%%%%%%%%%%%%%%%%%%%%%
for whole_iteration_k=1:15
    
    
    for u1=1:iteration_k1
        for u=1:1
            J_delta=2*C*WA_1+ 2*dataA_1do'*(dataA_1do*WA_1-dataB_1do*WB_1)-2*dataA_1do'*(dataA_1do*WA_1_t0-dataBwrong_1do*WB_1);
            
            WA_1=WA_1-leaning_rate1*J_delta;
        end
        WA_1_t0=WA_1;
    end
    W.WA_1=WA_1;
    
    [result(c)]=preded(W, weight, test1,test2,topk)
    c=c+1;
    
    for u1=1:iteration_k1
        for u=1:1
            J_delta=2*C*WA_2+ 2*dataA_2do'*(dataA_2do*WA_2-dataB_2do*WB_2)-2*dataA_2do'*(dataA_2do*WA_2_t0-dataBwrong_2do*WB_2);
            
            WA_2=WA_2-leaning_rate1*J_delta;
        end
        WA_2_t0=WA_2;
    end
    W.WA_2=WA_2;
    
    [result(c)]=preded(W, weight, test1,test2,topk)
    c=c+1;
    
    for u1=1:iteration_k1
        for u=1:1
            J_delta=2*C*WA_3+ 2*dataA_3do'*(dataA_3do*WA_3-dataB_3do*WB_3)-2*dataA_3do'*(dataA_3do*WA_3_t0-dataBwrong_3do*WB_3);
            
            WA_3=WA_3-leaning_rate1*J_delta;
        end
        WA_3_t0=WA_3;
    end
    W.WA_3=WA_3;
    
    [result(c)]=preded(W, weight, test1,test2,topk)
    c=c+1;
    
    for u1=1:iteration_k1
        for u=1:1
            J_delta=2*C*WA_4+ 2*dataA_4do'*(dataA_4do*WA_4-dataB_4do*WB_4)-2*dataA_4do'*(dataA_4do*WA_4_t0-dataBwrong_4do*WB_4);
            
            WA_4=WA_4-leaning_rate1*J_delta;
        end
        WA_4_t0=WA_4;
    end
    W.WA_4=WA_4;
    
    [result(c)]=preded(W, weight, test1,test2,topk)
    c=c+1;
    
    for u1=1:iteration_k1
        for u=1:1
            J_delta=2*C*WA_5+ 2*dataA_5do'*(dataA_5do*WA_5-dataB_5do*WB_5)-2*dataA_5do'*(dataA_5do*WA_5_t0-dataBwrong_5do*WB_5);
            
            WA_5=WA_5-leaning_rate1*J_delta;
        end
        WA_5_t0=WA_5;
    end
    W.WA_5=WA_5;
    
    [result(c)]=preded(W, weight, test1,test2,topk)
    c=c+1;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for u1=1:iteration_k2
        for u=1:1
            J_delta=2*100*WB_1-2*dataB_1do'*(dataA_1do*WA_1-dataB_1do*WB_1)+2*dataBwrong_1do'*(dataA_1do*WA_1-dataBwrong_1do*WB_1);
            WB_1=WB_1-leaning_rate2*J_delta;
        end
        WB_1_t0=WB_1;
    end
    W.WB_1=WB_1;
    
    [result(c)]=preded(W, weight, test1,test2,topk)
    c=c+1;
    
    for u1=1:iteration_k2
        for u=1:1
            J_delta=2*100*WB_2-2*dataB_2do'*(dataA_2do*WA_2-dataB_2do*WB_2)+2*dataBwrong_2do'*(dataA_2do*WA_2-dataBwrong_2do*WB_2);
            WB_2=WB_2-leaning_rate2*J_delta;
        end
        WB_2_t0=WB_2;
    end
    W.WB_2=WB_2;
    
    [result(c)]=preded(W, weight, test1,test2,topk)
    c=c+1;
    
    for u1=1:iteration_k2
        for u=1:1
            J_delta=2*100*WB_3-2*dataB_3do'*(dataA_3do*WA_3-dataB_3do*WB_3)+2*dataBwrong_3do'*(dataA_3do*WA_3-dataBwrong_3do*WB_3);
            WB_3=WB_3-leaning_rate2*J_delta;
        end
        WB_3_t0=WB_3;
    end
    W.WB_3=WB_3;
    
    [result(c)]=preded(W, weight, test1,test2,topk)
    c=c+1;
    
    for u1=1:iteration_k2
        for u=1:1
            J_delta=2*100*WB_4-2*dataB_4do'*(dataA_4do*WA_4-dataB_4do*WB_4)+2*dataBwrong_4do'*(dataA_4do*WA_4-dataBwrong_4do*WB_4);
            WB_4=WB_4-leaning_rate2*J_delta;
        end
        WB_4_t0=WB_4;
    end
    W.WB_4=WB_4;
    
    [result(c)]=preded(W, weight, test1,test2,topk)
    c=c+1;
    
    for u1=1:iteration_k2
        for u=1:1
            J_delta=2*100*WB_5-2*dataB_5do'*(dataA_5do*WA_5-dataB_5do*WB_5)+2*dataBwrong_5do'*(dataA_5do*WA_5-dataBwrong_5do*WB_5);
            WB_5=WB_5-leaning_rate2*J_delta;
        end
        WB_5_t0=WB_5;
    end
    W.WB_5=WB_5;
    
    [result(c)]=preded(W, weight, test1,test2,topk)
    c=c+1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    zty_1mm=sum(power(dataA_1do*WA_1-dataB_1do*WB_1,2),2);
    zty_1mmu=sum(power(dataA_1do*WA_1-dataBwrong_1do*WB_1,2),2);
    try_1=sum((zty_1mm-zty_1mmu));
        
    zty_2mm=sum(power(dataA_2do*WA_2-dataB_2do*WB_2,2),2);
    zty_2mmu=sum(power(dataA_2do*WA_2-dataBwrong_2do*WB_2,2),2);
    try_2=sum((zty_2mm-zty_2mmu));
    
    zty_3mm=sum(power((dataA_3do+1)*WA_3-dataB_3do*WB_3,2),2);
    zty_3mmu=sum(power((dataA_3do)*WA_3-dataBwrong_3do*WB_3,2),2);
    try_3=sum((zty_3mm-zty_3mmu));
    
    zty_4mm=sum(power((dataA_4do)*WA_4-dataB_4do*WB_4,2),2);
    zty_4mmu=sum(power((dataA_4do)*WA_4-dataBwrong_4do*WB_4,2),2);
    try_4=sum((zty_4mm-zty_4mmu));
    
    zty_5mm=sum(power((dataA_5do)*WA_5-dataB_5do*WB_5,2),2);
    zty_5mmu=sum(power((dataA_5do)*WA_5-dataBwrong_5do*WB_5,2),2);
    try_5=sum((zty_5mm-zty_5mmu));
    
    cvx_begin
    cvx_precision low
    cvx_solver MOSEK
    variables alp(5)
    minimize (norm(alp,1))
    
    subject to
    alp(1)*zty_1mm+alp(2)*zty_2mm+alp(3)*zty_3mm+alp(4)*zty_4mm+alp(5)*zty_5mm<=alp(1)*zty_1mmu+alp(2)*zty_2mmu+alp(3)*zty_3mmu+alp(4)*zty_4mmu+alp(5)*zty_5mmu
    sum(alp)==1
%     alp>0
    cvx_end
    
    alp(find(alp<0))=0;
    weight.alp=alp;
    result(c)=preded(W, weight, test1,test2,topk);
    c=c+1;
    
    
    weightchange(whole_iteration_k+1,:)=alp;
    
     
end





