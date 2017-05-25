% load('celebrity2000_meta.mat');
% for i = 1:22000
%     filename = celebrityImageData.name{i};
%     index = strfind(filename,'å');
%     if index
%         celebrityImageData.name{i}(index)='氓';
%     end
%     index = strfind(filename,'á');
%     if index
%         celebrityImageData.name{i}(index)='谩';
%     end
% end
% train_index = 1;
% test_index = 1;
% h = figure;
% for i = 1:22000
%     filename = ['./CACD2000/',celebrityImageData.name{i}];
%     im = imread(filename);
%     im = double(im);
%     im = im/max(max(max(im)));
%     label_index = celebrityImageData.identity(i);
%     label = zeros(1,301);
%     label(label_index)=1;
%     if mod(i,11)
%         train_path = sprintf('./train/%d.mat',train_index);
%         train_index = train_index+1;
%        % save(train_path, 'im','label');
%         subplot(2,10,mod(i,11))
%         imshow(im);
%     else
%         test_path = sprintf('./test/%d.mat',test_index);
%         test_index = test_index+1
%        % save(test_path,'im','label');
%         subplot(2,10,11)
%         imshow(im);
%         pause()
%         close(h);
%         h = figure;
%     end
% end


%% read images in CACD, and save as mat files
load ('cele.mat')
total_num = 163446
hwait = waitbar(0,'formating mat files');
for i = 1:total_num
    str=['formating mat files:   ',num2str(i/total_num*100),'%'];
    waitbar(i/total_num,hwait,str);
    filename = ['../../dataset/CACD2000/',celebrityImageData.name{i}];
    im = imread(filename);
    im = double(im);
    im = im/max(max(max(im)));
    label=celebrityImageData.identity(i)-1;
    path = sprintf('../data/%d.mat',i);
    save(path, 'im','label','-v6');
end
%%


    