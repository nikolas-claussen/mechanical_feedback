%% Sebastian's cytosolic normalization script
% 08/20/2021. Normalize by using a tophat filter.


%% dict of background values (outside of embryo in 3d volume)
% not nec for fat2, background basically 0 for some reason

%root_path = '/home/nikolas/hdd/hdd/data/streichan/myosin data/Hannah''s data/twist_control/';
%save_path = strcat(root_path, 'sebastian_processed/');
%embryos = [202006261115, 202007081130 202008131005 202007011145 202007091200 202009041145];
%min_vals = containers.Map({202006261115 202007081130 202008131005 202007011145 202007091200 202009041145}, {1860 2390 560 1100 830 640});

%root_path = '/home/nikolas/hdd/hdd/data/streichan/myosin data/Hannah''s data/twist_minus/';
%save_path = strcat(root_path, 'cytosolic/');
%embryos = [202007171100, 202007301030, 202007171400, 202007301145, 202007231115, 202008121100];
%min_vals = containers.Map({202007171100, 202007301030, 202007171400, 202007301145, 202007231115, 202008121100},...
%    {1530 560 1050 1390 530 970});

%root_path = '/home/nikolas/hdd/hdd/data/streichan/atlas data/mutants/Even-Skipped[r13]/Spaghetti_Squash-GFP/2020 datasets/';
%save_path = strcat(root_path, 'cytosolic/');
%embryos = [202001151953, 202001181115];
%min_vals = containers.Map({202001151953, 202001181115},...
%    {100 150});

%root_path = '/home/nikolas/Documents/UCSB/streichan/Matt_paper/runt myo processing/results/202109081145/';
%save_path = strcat(root_path, '/cytosolic/'); %strcat(root_path, 'eve R13 cytosolic/2020_datasets/');
%embryos = ["202109081145"];
%min_vals = containers.Map({202109081145}, {0});
% 1200 for 2020 datasets 

% PMG normalization, obsolete
%root_path = '/home/nikolas/hdd/hdd/data/streichan/myosin data/Hannah''s data/twist_control/';
%save_path = strcat(root_path, 'WT cytosolic/');
%embryos = [202006261115, 202007091200, 202007011145, 202009041145, 202007081130, 202008131005];
%min_vals = containers.Map({202006261115 202007091200 202007011145 202009041145 202007081130 202008131005},...
%    {6093 4510 2418 2823 4585 2306});

%root_path = '/home/nikolas/hdd/hdd/data/streichan/myosin data/eve/Spaghetti_Squash-GFP/';
%save_path = strcat(root_path, 'eve R13 cytosolic/');
%embryos = [202101231300, 202103141755, 202103142135];
%min_vals = containers.Map({202101231300 202103141755 202103142135}, {118 32 16});

%root_path = '/home/nikolas/hdd/hdd/data/streichan/myosin data/eve/Spaghetti_Squash-GFP/';
%save_path = strcat(root_path, 'cytosolic/');
%embryos = [202107071520, 202107071645, 202103141755];
%min_vals = containers.Map({202107071520, 202107071645, 202103141755},...
%    {20 30 20});

root_path = '/home/nikolas/hdd/hdd/data/streichan/myosin data/eve hetero/Halo_Hetero_Eve_Hetero/';
save_path = root_path;
embryos = [202001151210, 202001152203, 202001162129, 202001171032, 202001181005,...
    202001211009, 202001151535, 202001161835, 202001162311, 202001171200, 202001181245];


 

cell_size = 8;
fill_holes = false; % use this to fill nuclei/fold induced holes in background before dividing by it
smooth_background = false; % smooth background slightly (probably no matter)

%% load images

for embryo = embryos
    disp(num2str(embryo))

    %Name = strcat(root_path, num2str(embryo), '/', dir(strcat(root_path, num2str(embryo), '/*.tif')).name);
    Name = strcat(root_path, num2str(embryo), '/', 'cylinder1_max', '.tif');

    % to overwrite, delete previous results
    ResultName = strcat(save_path, num2str(embryo), '/', num2str(embryo), '_cytosolic_prelim', '.tif');
    if exist(ResultName) == 2
        delete(ResultName);
    end
    
    StackSize = length(imfinfo(Name));
    Size = size(imread(Name, 1));
    for k = 1 :  StackSize 
        image = double(imread(Name,k));
        %image = double(image-min_vals(embryo)); % add clip step ? 
        % background is a (smoothed) top-hat transform of the original
        background = imdilate(imerode(image,strel('disk',cell_size)), strel('disk',cell_size));
        if smooth_background
            background = imgaussfilt(background, cell_size/2); 
        end
        if fill_holes
            background_filled = imfill(background);
            normalized = (image-background)./background_filled;
        else
            normalized = (image-background)./background;
        end
        % save
        normalized = uint16(1000 * normalized);
        imwrite(normalized, ResultName, 'tiff', 'Compression', 'none', 'WriteMode', 'append');
    end
end 











