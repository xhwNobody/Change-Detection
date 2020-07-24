pic1 = 'C:\Users\12271\Desktop\change-detection\dataset\Ottawa\199707.png';
pic2 = 'C:\Users\12271\Desktop\change-detection\dataset\Ottawa\199708.png';
pic1data = imread(pic1);
pic2data = imread(pic2);
pic1data = double(pic1data(:,:,1));
pic2data = double(pic2data(:,:,1));
picsize = size(pic1data);
similarity = zeros(picsize);
for i=1:picsize(1)
    for j=1:picsize(2)
        temp = pic1data(i,j) + pic2data(i,j);
        if temp ~= 0
            similarity(i,j) = (abs(pic1data(i,j) - pic2data(i,j)))/temp;
        else
            similarity(i,j) = 0;
        end
    end
end
similarity = similarity(:);
%分成2类
opti = [2.0;100000.0;0.0001;0];
[center,U,obj_fcn] = fcm(similarity,2,opti);
[~,label] = max(U); %找到所属的类
%变化到图像的大小
img_new = reshape(label,picsize);
imshow(img_new,[])

