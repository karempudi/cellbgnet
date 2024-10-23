%% Set path to Imanalysis
elnDir = '/home/pk/Documents/3DForkPlots/chromosome/';
addpath([elnDir '/matlabcode/']);
addpath([elnDir '/ImAnalysis/']);
ImAnalysis_setup([elnDir '/ImAnalysis/']);
%% Load raw data and convert to the pipeline format
pxSize=65; %nm

% 'molecule','image','x','y','z','photon','prob',
% 'sigma_x','sigma_y','sigma_z','sigma_photon','x_offset','y_offset','z_offset'
coords=table2array(readtable(fullfile(elnDir,'decode_train_output_prob_filtered.csv')));
%coords=coords(:,2:10);
coords=coords(:, 2:10);

coords(:,2:3)=coords(:,2:3)/pxSize+1; % python counts from 0, MATLAB from 1.

% 'molecule','image','x','y','z','photon','prob'
%trueCoords=table2array(readtable(fullfile(elnDir,'chromosome/train_data.csv')));
%trueCoords=trueCoords(:,2:6);
%trueCoords(:,2:3)=trueCoords(:,2:3)/pxSize+1;



%% Create a directory with soft links to training cell masks
% T=readtable(fullfile(elnDir,'train_cell_masks.txt'),'Delimiter','','ReadVariableNames',false);
% filenames=table2cell(T);
% filenames = split(filenames,'/');
% filenames = filenames(:,end);
% sourceDir='/mnt/sda1/SMLAT/data/real_data/chromosome_dots/pooled_8865/phase_venus_mask/';
% targetDir=fullfile(elnDir,'train_cell_masks');
% if isfolder(targetDir)==0
%     mkdir(targetDir)
%     for i=1:numel(filenames)
%         sourcefile=fullfile(sourceDir,filenames{i});
%         targetfile=fullfile(targetDir,[num2str(i-1,'%04d') '.png']);
%         system(sprintf('ln -s %s %s',sourcefile,targetfile));
%     end
% end
%% Compute cell measurements and internal dot coordinates
segDir = fullfile(elnDir,'train_cell_masks/');
mCells=computeMCellsAndInternalDotCoordinates(coords,segDir,'venus');

mCellsFile = fullfile(elnDir,'mCells_prob_filtered.mat');
Cell.MCell.saveMCells(mCellsFile,mCells);
%% Make forkplots
lengthCutOff = 1;
parentLengthCutOff = 0;
daughterLengthCutOff = 0;
estimateInitArea = 0; 

[shorts, longs, areas, lengths, widths, counts, depths] =  ...
    getDotLocations3D(mCells, 'lengthCutOff', lengthCutOff, ...
    'parentLengthCutoff', parentLengthCutOff, 'daughterLengthCutoff', daughterLengthCutOff, ...
    'orientFlag', 0, 'PixelSize',pxSize/1000);

birthAreas = [];
divisionAreas=[];
initiationAreas=[];

noNaNs = find(~isnan(longs));
displayForkPlot(longs(noNaNs), areas(noNaNs), lengths(noNaNs), counts(noNaNs), ...
    mean(birthAreas), mean(divisionAreas), mean(initiationAreas), 'BinScale', 15, 'HeatMapThreshold',0.95);
xlim([-3 3])
ylim([0.8 2.4])

displayForkPlotShortAxis(shorts(noNaNs), areas(noNaNs), widths(noNaNs), counts(noNaNs), ...
    mean(birthAreas), mean(divisionAreas), mean(initiationAreas), 'BinScale', 80, 'HeatMapThreshold',0.9);
% xlim([-0.5 0.5])
% ylim([1 3])

displayForkPlotDepthAxis(depths(noNaNs), areas(noNaNs), widths(noNaNs), counts(noNaNs), ...
   mean(birthAreas), mean(divisionAreas), mean(initiationAreas), 'BinScale', 80,'HeatMapThreshold',0.9);
% xlim([-0.5 0.5])
% ylim([1 3])
%% Plot some histograms
figure, hold on
histogram(coords(:,4))
ylabel('counts');
xlabel('z [nm]');

figure, hold on
histogram(coords(:,5))
ylabel('counts');
xlabel('photon count');

dots = getParticleList(mCells);
y = dots.internalCoord(:, 2);

z = dots.fluoCoord(:, 3);
figure, hold on
histogram2(y, z, -1:0.05:1, -500:25:500,'DisplayStyle', 'tile')
xlabel('Y internal coordinate')
ylabel('z [nm]')
% imDir=fullfile(elnDir,'chromosome/train_images/');
% segDir=fullfile(elnDir,'chromosome/train_cell_masks/');
% frame=1;
% im=imread(fullfile(imDir,[num2str(frame-1,'%04d') '.tiff']));
% figure,imshow(im,[90 200])
% hold on
% % plot(coords(ind,2),coords(ind,3),'rx')
% % ind=trueCoords(:,1)==frame;
% % plot(trueCoords(ind,2),trueCoords(ind,3),'bx')
% segIm=imread(fullfile(segDir,[num2str(frame-1,'%04d') '.png']));
% visboundaries(bwboundaries(segIm>0,'noholes'),'EnhanceVisibility',0,'Color','g','LineWidth',1);
% ind=coords(:,1)==frame;
% viscircles(coords(ind,2:3),1,'Color','r','EnhanceVisibility',0,'LineWidth',1);
% ind=trueCoords(:,1)==frame;
% viscircles(trueCoords(ind,2:3),1,'Color','b','EnhanceVisibility',0,'LineWidth',1);
