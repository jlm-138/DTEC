function [tap, membership] = DTEC(data)
%% DTEC: Decision Tree-based Evidential Clustering
% [tap, membership] = DTEC(data)
% Input: 
%    - data: data needs clustering, where rows represent instances, columns
%    represent features
% Output: 
%    - tap: cluster label 
%    - membership: evidential membership to clusters 
% Version 1.0 -- 2023/3/21
[a,b]=size(data);
y=zeros(a,1);
At_Max = max(data(:,1:b));
At_Min = min(data(:,1:b));
for i = 1:a
    data(i,1:b) = (data(i,1:b) - At_Min)./(At_Max-At_Min);
end%归一�?
% scatter3(data(:,1),data(:,2),data(:,3),'k');
distance=zeros(a,a);%计算两两样本间距离，用于后续轮廓度量的计�?
for i=1:a-1%a为样本数
    for j=i+1:a
        for k=1:b%b是特征维�?
            distance(i,j)=distance(i,j)+(data(i,k)-data(j,k))^2;
        end
        distance(i,j)=distance(i,j)^0.5;
        distance(j,i)=distance(i,j);
    end
end
Data=[data y ones(a,1)];
for i=1:a
    Data(i,b+1)=i;
end
Index=["1";""];%记录节点类型，数字表示其中含有的类别个数
Cut=zeros(1,5);
C=zeros(1,3);
g=1;
OBJ=0;
spc=5;
str='a';
B=1;
while g<=B
    flag=1;
    if Index(1,g)=="1"
        posi=find(Index(1,:)=="1");
        [~,numb]=size(posi);
        da=Data(:,b+1+posi);
        [~,x]=size(Data);
        Posi=find(Index(1,:)=="2");
        [~,Numb]=size(Posi);
        if Numb~=0
            for i=1:Numb
                for j=1:numb
                    if contains(Index(2,Posi(1,i)),Index(2,posi(1,j)))==1
                        da(:,j)=da(:,j)+Data(:,b+1+Posi(1,i))/2;
                    end
                end
            end
        end
        Data=[Data da];
        numb=numb+1;
        center=zeros(numb+1,b);
        [~,index]=max(da,[],2);
        for i=1:numb-1
            if posi(1,i)==g
                continue;
            end
            ma=find(index==i);
            center(i,:)=sum(Data(ma,1:b).*Data(ma,b+1+posi(1,i)),1)/sum(Data(ma,b+1+posi(1,i)));
        end
        [~,v]=size(find(Index(1,1:g)=="1"));
        for i=1:b
            [Data,D]=sortrows(Data,i);
            distance=distance(D,:);
            distance=distance(:,D);
            da=Data(:,x+1:end);
            temp=sort(unique(Data(:,i)));
            [w,~]=size(temp);
            cut=zeros(w-1,1);
            for j=1:w-1
                cut(j,1)=(temp(j,1)+temp(j+1,1))/2;
            end
            [c,~]=size(cut);
            Z=find(i==Cut(:,3));
            if isempty(Z)==1
                nu=0;
            else
                [~,nu]=size(Z);
            end
            bucket=zeros(1,2);
            for j=1:c
                if nu~=0
                    Flag=1;
                    for k=1:nu
                        if cut(j,1)<=Cut(Z(1,k),5)&&cut(j,1)>=Cut(Z(1,k),4)
                            Flag=0;
                            break;
                        end
                    end
                    if Flag==0
                        continue;
                    end
                end
                d=find(Data(:,i)>cut(j,1));
                d=d(1,1)-1;
                Data=[Data zeros(a,2)];
                Data(1:d,x+numb)=Data(1:d,x+v);
                Data(d+1:end,x+numb+1)=Data(d+1:end,x+v);
                Data(:,x+v)=zeros(a,1);
                [~,H]=max(Data(:,x+1:end),[],2);
                [bucket(1,1),~]=size(find(H==numb));
                [bucket(1,2),~]=size(find(H==numb+1));
                lk=zeros(a,4);
                [~,p]=max(Data(:,x+1:end),[],2);
                E=tabulate(p);
                [Y,~]=size(find(E(:,2)>1));
                if isempty(p(E(:,2)==1))==0||Y<numb
                    Data=Data(:,1:x);
                    Data=[Data da];
                    continue;
                end
                if min(bucket)<=spc
                    Data=Data(:,1:x);
                    Data=[Data da];
                    continue;
                end
                center(numb,:)=sum(Data(:,1:b).*Data(:,x+numb),1)/sum(Data(:,x+numb));
                center(numb+1,:)=sum(Data(:,1:b).*Data(:,x+numb+1),1)/sum(Data(:,x+numb+1));
                closest=ones(numb+1,numb+1).*b;
                for k=1:numb
                    if k<=numb-1&&posi(1,k)==g
                        continue;
                    end
                    for m=k+1:numb+1
                        if m<=numb-1&&posi(1,m)==g
                            continue;
                        end
                        closest(k,m)=sqrt(sum((center(k,:)-center(m,:)).^2));
                        closest(m,k)=closest(k,m);
                    end
                end
                clo=zeros(1,numb+1);
                for k=1:numb+1
                    if k<=numb-1&&posi(1,k)==g
                        continue;
                    end
                    [~,clo(1,k)]=min(closest(k,:));
                end
                sm=zeros(a,1);
                for k=1:a-1
                    for m=k+1:a
                        if k==m
                            continue;
                        end
                        if p(k,1)==p(m,1)
                            lk(k,1)=lk(k,1)+distance(k,m)*Data(m,x+p(m,1));
                            lk(k,2)=lk(k,2)+Data(m,x+p(m,1));
                            lk(m,1)=lk(m,1)+distance(k,m)*Data(k,x+p(k,1));
                            lk(m,2)=lk(m,2)+Data(k,x+p(k,1));
                        elseif clo(1,p(k,1))==p(m,1)
                            lk(k,3)=lk(k,3)+distance(k,m)*Data(m,x+p(m,1));
                            lk(k,4)=lk(k,4)+Data(m,x+p(m,1));
                        end
                        if clo(1,p(m,1))==p(k,1)
                            lk(m,3)=lk(m,3)+distance(k,m)*Data(k,x+p(k,1));
                            lk(m,4)=lk(m,4)+Data(k,x+p(k,1));
                        end
                    end
                    sm(k,1)=(lk(k,3)/lk(k,4)-lk(k,1)/lk(k,2))/max(lk(k,3)/lk(k,4),lk(k,1)/lk(k,2));
                end
                sm(a,1)=(lk(a,3)/lk(a,4)-lk(a,1)/lk(a,2))/max(lk(a,3)/lk(a,4),lk(a,1)/lk(a,2));
                obj=sum(sm);
                if obj>OBJ
                    flag=0;
                    C(g,1)=d;
                    C(g,2)=i;
                    C(g,3)=cut(j,1);
                    OBJ=obj;
                end
                Data=Data(:,1:x);
                Data=[Data da];
            end
        end
        Data=Data(:,1:x);
        if flag==1
            Cut(g,:)=zeros(1,5);
            C(g,:)=zeros(1,3);
            g=g+1;
            continue;
        end
        [Data,D]=sortrows(Data,C(g,2));
        distance=distance(D,:);
        distance=distance(:,D);
        center=zeros(3,b);
        center(1,:)=sum(Data(1:C(g,1),1:b).*Data(1:C(g,1),b+1+g),1)/sum(Data(1:C(g,1),b+1+g));
        center(2,:)=sum(Data(C(g,1)+1:end,1:b).*Data(C(g,1)+1:end,b+1+g),1)/sum(Data(C(g,1)+1:end,b+1+g));
        center(3,C(g,2))=C(g,3);%得到左右两侧及交集簇的簇�?
        belif=zeros(a,3);
        for i=1:a
            dis=zeros(1,3);
            dis(1,1)=(Data(i,C(g,2))-center(1,C(g,2)))^2;
            dis(1,2)=(Data(i,C(g,2))-center(2,C(g,2)))^2;
            dis(1,3)=(Data(i,C(g,2))-center(3,C(g,2)))^2;
            belif(i,1)=dis(1,2)*dis(1,3)/(dis(1,2)*dis(1,1)+dis(1,1)*dis(1,3)+dis(1,2)*dis(1,3));
            belif(i,2)=dis(1,1)*dis(1,3)/(dis(1,2)*dis(1,1)+dis(1,1)*dis(1,3)+dis(1,2)*dis(1,3));
            belif(i,3)=dis(1,1)*dis(1,2)/(dis(1,2)*dis(1,1)+dis(1,1)*dis(1,3)+dis(1,2)*dis(1,3));
            belif(i,3)=belif(i,3)*Data(i,b+1+g);
            dis(1,1)=((Data(i,C(g,2))-center(1,C(g,2)))/(C(g,3)-center(1,C(g,2))))^2;
            dis(1,2)=((Data(i,C(g,2))-center(2,C(g,2)))/(C(g,3)-center(2,C(g,2))))^2;
            belif(i,1)=(Data(i,b+1+g)-belif(i,3))*dis(1,2)/(dis(1,1)+dis(1,2));
            belif(i,2)=(Data(i,b+1+g)-belif(i,3))*dis(1,1)/(dis(1,1)+dis(1,2));
        end
        [~, loc] = max(belif,[],2);%记录�?���?
        [~,IA]=unique(loc);
        IA=sort(IA);
        [J,~]=size(IA);
        Cut(g,1)=IA(2,1);
        Cut(g,3)=C(g,2);
        Cut(g,4)=(Data(IA(2,1)-1,C(g,2))+Data(IA(2,1),C(g,2)))/2;
        if J==2
            Cut(g,2)=IA(2,1);
            Cut(g,5)=(Data(IA(2,1)-1,C(g,2))+Data(IA(2,1),C(g,2)))/2;
        else
            Cut(g,2)=IA(3,1);
            Cut(g,5)=(Data(IA(3,1)-1,C(g,2))+Data(IA(3,1),C(g,2)))/2;
        end
        str=char(str);
        Index=[Index(1,:) "1" "1" "2";Index(2,:) str char(str+1) [str,str+1]];
        if g~=1
            for i=1:Numb
                Index(2,Posi(1,i))=strrep(Index(2,Posi(1,i)),Index(2,g),[str,str+1]);
            end
        end
        str=str+2;
        Index(1,g)="0";
        Data=Data(:,1:x);
        Data=[Data belif];
    elseif Index(1,g)=="2"
        h=strlength(Index(2,g));
        if h<3
            g=g+1;
            continue;
        else
            [~,B]=size(Index);
            zifu=Index(2,find(strlength(Index(2,:))==1));
            [~,len]=size(zifu);
            J=zeros(1,len);
            for i=1:len
                J(1,i)=contains(Index(2,g),zifu(1,i));
                J(1,i)=J(1,i)*i;
            end
            J(J==0) = [];
            J=unique(ceil(J/2));
            f=C;
            f(all(f  == 0,2),:) = [];
            CUT=f(J(1,1),:);
            Data=sortrows(Data,CUT(1,2));
            Posi=find(Index(1,:)=="2");
            [~,Numb]=size(Posi);
            G=B;
            for i=1:Numb
                if Posi(1,i)==g
                    continue;
                end
                if G~=B&&contains(Index(2,g),Index(2,Posi(1,i)))==1
                    Str=char(Index(2,Posi(1,i)-1));
                    Index=[Index(1,:) "2" "2";Index(2,:) "" ""];
                    Index(2,G+1)=strrep(Index(2,G),Str,"");
                    Index(2,G+2)=strrep(Index(2,G),char(Str-1),"");
                    Index(2,G-1)=strrep(Index(2,G-1),Str,"");
                    Index(2,G)=strrep(Index(2,G-1),char(Str-1),Str);
                    G=G+2;
                elseif contains(Index(2,g),Index(2,Posi(1,i)))==1
                    Str=char(Index(2,Posi(1,i)-1));
                    Index=[Index(1,:) "2" "2";Index(2,:) "" ""];
                    Index(2,G+1)=strrep(Index(2,g),Str,"");
                    Index(2,G+2)=strrep(Index(2,g),char(Str-1),"");
                    G=G+2;
                end
            end
            belif=zeros(a,2);
            if sum(Data(1:CUT(1,1),b+1+g))==0
                belif(:,1)=zeros(a,1);
                belif(:,2)=Data(:,b+1+g);
            elseif sum(Data(CUT(1,1)+1:end,b+1+g))==0
                belif(:,1)=Data(:,b+1+g);
                belif(:,2)=zeros(a,1);
            else
                cen1=sum(Data(1:CUT(1,1),1:b).*Data(1:CUT(1,1),b+1+g),1)/sum(Data(1:CUT(1,1),b+1+g));
                cen2=sum(Data(CUT(1,1)+1:end,1:b).*Data(CUT(1,1)+1:end,b+1+g),1)/sum(Data(CUT(1,1)+1:end,b+1+g));
                for i=1:a
                    dis=zeros(1,2);
                    dis(1,1)=((Data(i,CUT(1,2))-cen1(1,CUT(1,2)))/(CUT(1,3)-cen1(1,CUT(1,2))))^2;
                    dis(1,2)=((Data(i,CUT(1,2))-cen2(1,CUT(1,2)))/(CUT(1,3)-cen2(1,CUT(1,2))))^2;
                    belif(i,1)=dis(1,2)/(dis(1,1)+dis(1,2))*Data(i,b+1+g);
                    belif(i,2)=dis(1,1)/(dis(1,1)+dis(1,2))*Data(i,b+1+g);
                end
            end
            Data=[Data belif];
            Index(1,g)="0";
            if h==4
                CUT=f(J(1,2),:);
                Data=sortrows(Data,CUT(1,2));
                [~,B]=size(Index);
                for j=1:2
                    belif=zeros(a,2);
                    if sum(Data(1:CUT(1,1),b+1+B-4+j))==0
                        belif(:,1)=zeros(a,1);
                        belif(:,2)=Data(:,b+1+B-4+j);
                    elseif sum(Data(CUT(1,1)+1:end,b+1+B-4+j))==0
                        belif(:,1)=Data(:,b+1+B-4+j);
                        belif(:,2)=zeros(a,1);
                    else
                        cen1=sum(Data(1:CUT(1,1),1:b).*Data(1:CUT(1,1),b+1+B-4+j),1)/sum(Data(1:CUT(1,1),b+1+B-4+j));
                        cen2=sum(Data(CUT(1,1)+1:end,1:b).*Data(CUT(1,1)+1:end,b+1+B-4+j),1)/sum(Data(CUT(1,1)+1:end,b+1+B-4+j));
                        for k=1:a
                            dis=zeros(1,2);
                            dis(1,1)=((Data(k,CUT(1,2))-cen1(1,CUT(1,2)))/(CUT(1,3)-cen1(1,CUT(1,2))))^2;
                            dis(1,2)=((Data(k,CUT(1,2))-cen2(1,CUT(1,2)))/(CUT(1,3)-cen2(1,CUT(1,2))))^2;
                            belif(k,1)=dis(1,2)/(dis(1,1)+dis(1,2))*Data(k,b+1+B-4+j);
                            belif(k,2)=dis(1,1)/(dis(1,1)+dis(1,2))*Data(k,b+1+B-4+j);
                        end
                    end
                    Data=[Data belif];
                end
                Data(:,b+2+B-4:b+3+B-4)=[];
            end
        end
    end
    g=g+1;
    [~,B]=size(Index);
end
z=find(Index(1,:)~="0");
Data=sortrows(Data,b+1);
membership=Data(:,z+b+1);
tap=Index(2,z);
end