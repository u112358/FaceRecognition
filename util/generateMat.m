%% read images in CACD, and save as mat files
total_num = 163446
cnt = 1
hwait = waitbar(0,'formating mat files');
classes = dir('./Processed_Aligned/');
for i = 3:size(classes,1)

    subdir = [classes(i).folder,'/',classes(i).name];
    instances = dir(subdir);

    for j=3:size(instances,1)
        filename = [subdir,'/',instances(j).name];
        im = imread(filename);
        im = double(im);
        m = mean(mean(mean(im)));
        ad_std = max(std(reshape(im,[250*250*3,1])),1.0/sqrt(250*250*3));
        im = (im-m)/ad_std;
        label=str2num(classes(i).name)-1;
        path = sprintf('./final/%d.mat',cnt);
        save(path, 'im','label','-v6');
        str=['formating mat files:   ',num2str(cnt/total_num*100),'%'];
        waitbar(cnt/total_num,hwait,str);
        cnt=cnt+1;
    end
end
%%  