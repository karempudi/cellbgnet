

pixelSize = 3.45/100;

analysisPath = '/crex/proj/uppstore2018129/elflab/Projects/Chromosome_structure/EXP-22-BY8866';
pipelineOutputPath = [analysisPath '/output'];
expInfoObj = loadExpInfo(pipelineOutputPath,'expInfoObj.mat');
posList = expInfoObj.getPositionList();
fluoChanNames = expInfoObj.getChannelNames('fluo');
fluoChanName = fluoChanNames{1};

nPositions = length(posList);
firstStrainPos = 1:length(posList)/2;
secondStrainPos = setdiff(1:length(posList), firstStrainPos);
strainPos = {firstStrainPos secondStrainPos};

lengthCutOff = 20;
parentLengthCutOff = 10;
daughterLengthCutOff = 10;
estimateInitArea = 0; 

if estimateInitArea
    sisterLengthCutOff = 3;
    N = 1;%number of generations
    useParent = 0;
    useDaughters = 1;
    minTrackLength = 10;
    debugFlag = 0;
    
    %Tracking params
    uParam.uWindow = 4;
    uParam.uMergeSplit = 1;
    uParam.uMinTrackLen = 1;
    uParam.uBrownStdMult = -1;
    uParam.uMinSearchRadius = 20;
    uParam.uMaxSearchRadius = 30;
    uParam.uLinearMotion = 0;
    uParam.uMinGapSearchRadius = 10;
    uParam.uMaxGapSearchRadius = 20;
    uParam.uAmpRatioLimit = [];
    uParam.uGapPenalty = 0.3;
end

for k = 1:length(strainPos)
    currStrainPos = strainPos{k};
    for j = 1:length(fluoChanNames)
        fluoChanName = fluoChanNames{j};
        shorts = [];
        longs = [];
        areas = [];
        lengths = [];
        widths = [];
        counts = [];
        depths = [];
        tmpBirthAreas = cell(1,nPositions);
        tmpDivisionAreas = cell(1,nPositions);
        tmpInitiationAreas = cell(1,nPositions);
        parfor i = currStrainPos
            %load(fullfile(cherryFolder, thePositions(i).name, 'trackedCells.mat'))
            matFile = expInfoObj.getMCellMatPath(posList{i});
            mCells = Cell.MCell.loadMCells(matFile);
            posName = posList{i};
            fluoIndices = expInfoObj.getIndices(posName,fluoChanName);

            [intCoordsShortAxis, intCoordsLongAxis, cellAreas, cellLengths, cellWidths, cellCounts, intCoordsDepthAxis] =  ...
                getDotLocations3D(mCells, 'lengthCutOff', lengthCutOff, ...
                'parentLengthCutoff', parentLengthCutOff, 'daughterLengthCutoff', daughterLengthCutOff, ...
                'orientFlag', 0, 'PixelSize',pixelSize, 'fluoChanName', fluoChanName);
            shorts = [shorts intCoordsShortAxis];
            longs = [longs intCoordsLongAxis];
            depths = [depths intCoordsDepthAxis];
            areas = [areas cellAreas];
            lengths = [lengths cellLengths];
            widths = [widths cellWidths];
            counts = [counts cellCounts];

            if estimateInitArea
                tmpInitiationAreas{i} = getInitiationAreas(mCells, fluoIndices, ...
                    N, lengthCutOff, parentLengthCutOff, daughterLengthCutOff, 0, ...
                    useParent, useDaughters, uParam, minTrackLength, pixelSize, 0, fluoChanName);%
            end

            tmpBirthAreas{i} = getBirthAreas(mCells,lengthCutOff,...
                parentLengthCutOff,daughterLengthCutOff,'PixelSize',pixelSize);

            tmpDivisionAreas{i} = getDivisionAreas(mCells,lengthCutOff,...
                parentLengthCutOff,daughterLengthCutOff,'PixelSize',pixelSize);


        end

        birthAreas = cell2mat(tmpBirthAreas);
        divisionAreas = cell2mat(tmpDivisionAreas);
        initiationAreas = cell2mat(tmpInitiationAreas);

        noNaNs = find(~isnan(longs));
        displayForkPlot(longs(noNaNs), areas(noNaNs), lengths(noNaNs), counts(noNaNs), ...
            mean(birthAreas), mean(divisionAreas), mean(initiationAreas), 'BinScale', 15); %, 'HeatMapThreshold',0.9
        xlim([-3.5 3.5])
        ylim([1 3])

        displayForkPlotShortAxis(shorts(noNaNs), areas(noNaNs), widths(noNaNs), counts(noNaNs), ...
            mean(birthAreas), mean(divisionAreas), mean(initiationAreas), 'BinScale', 80); %, 'HeatMapThreshold',0.9
        xlim([-0.5 0.5])
        ylim([1 3])

        displayForkPlotDepthAxis(depths(noNaNs), areas(noNaNs), widths(noNaNs), counts(noNaNs), ...
           mean(birthAreas), mean(divisionAreas), mean(initiationAreas), 'BinScale', 80); %, 'HeatMapThreshold',0.9
        xlim([-0.5 0.5])
        ylim([1 3])

    end

    figure; h = histogram(birthAreas, 50, 'Normalization', 'pdf');
    hold on
    histogram(divisionAreas, 'BinWidth', h.BinWidth, 'Normalization', 'pdf')
    hold off
    xlim([0.5 4])
    legend('Birth areas', 'Division areas')

    figure; histogram2(birthAreas, divisionAreas, 50, 'DisplayStyle','tile','Normalization','pdf','ShowEmptyBins','on')
    xlim([0.75 2])
    ylim([1.5 4])
    xlabel('Birth areas (µm^2)')
    ylabel('Division areas (µm^2)')
    hold on
    plot([0 2], [0 4], 'r', 'LineWidth', 2)
    hold off

end





