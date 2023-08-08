function displayForkPlotDepthAxis(varargin)
% Creates a fork distribution plot by binning cell areas and cell lengths
% and counting normalized number of dots per bin.
% 
% Input:
%   depths - array of dot internal coordinates along the depth axis
%   (z-axis)
%   areas - array of cell areas (in um^2)
%   widths - array of cell widths (in um)
%   counts - array of cell counts
%   avgBirthArea - average cell area at birth (in um^2).
%                  Optional, by default [].
%   avgDivisionArea - average cell area at division (in um^2).
%                     Optional, by default [].
%   avgInitiationArea - average cell area at initation (in um^2)
%                       Optional, by default [].
% 
% displayForkPlot(AX,...) plots into the axes with handle AX.
%
% displayForkPlot(...,'BinScale',binScale) adjusts the number of bins. The
% higher binScale, the narrower are bins. Default value is 20.
% 
% displayForkPlot(...,'HeatMapThreshold',P) uses P-quantile of the heatmap
% for its thresholding. If both average birth and division areas are
% provided, heatmap range is limited by these areas. Must be a scalar in 
% range (0,1], default value is 0.99.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse parameters
if length(varargin)>1 && isa(varargin{1},'matlab.graphics.axis.Axes')
    ax = varargin{1};
    varargin = varargin(2:end);
else
    ax = [];
end

ip = inputParser;
validNonEmpty = @(x) ~isempty(x) && isnumeric(x);
ip.addRequired('depths',validNonEmpty);
ip.addRequired('areas',validNonEmpty);
ip.addRequired('widths',validNonEmpty);
ip.addRequired('counts',validNonEmpty);
ip.addOptional('avgBirthArea',[]);
ip.addOptional('avgDivisionArea',[]);
ip.addOptional('avgInitiationArea',[]);
ip.addParameter('BinScale',20, @(x) isscalar(x) && isnumeric(x) && x>0);
ip.addParameter('HeatMapThreshold',0.99,@(x) isscalar(x) && x>0 && x<=1);

ip.parse(varargin{:});
areas = ip.Results.areas;
counts = ip.Results.counts;
depths = ip.Results.depths;
widths = ip.Results.widths;
avgBirthArea = ip.Results.avgBirthArea;
avgDivisionArea = ip.Results.avgDivisionArea;
avgInitiationArea = ip.Results.avgInitiationArea;
binScale = ip.Results.BinScale;
heatMapThresh = ip.Results.HeatMapThreshold;

%%%%%%%%%%%%%%%%%%%
% Bin cell areas and lengths and compute dot counts per bin

dmax = max(depths);
dmin = min(depths);
dminmax = max(dmax,-dmin);
dnrbins = round(2*dminmax*binScale);
if ~mod(dnrbins,2)
    dnrbins = dnrbins+1;
end
dbins = linspace(-dminmax, dminmax, dnrbins);

aminmax = quantile(areas, [0.005 0.98]);
amin = aminmax(1);
amax = aminmax(2);
anrbins = dnrbins;
abins = linspace(amin, amax, anrbins);

heatMap = zeros(anrbins - 1, dnrbins);
meanCellWidths = zeros(1, anrbins-1);
for i = 1:anrbins - 1
    selDots = areas > abins(i) & areas <= abins(i+1);
    selWidths = widths(selDots);
    selCounts = counts(selDots);
    selDepth = depths(selDots);
    h1 = hist(selDepth, dbins);
    normFactor = sum(1./selCounts);
    heatMap(i,:) = h1./normFactor;
    meanCellWidths(i) = mean(selWidths);   
end

if heatMapThresh < 1
    heatMap(heatMap==0) = nan;
    if ~isempty(avgBirthArea) && ~isempty(avgDivisionArea)
        ind = abins(1:end-1) >= avgBirthArea & abins(2:end) <= avgDivisionArea;
        data = heatMap(ind,:);
    else
        data = heatMap;
    end
    thresh = quantile(data(:),heatMapThresh);
    heatMap(heatMap>thresh) = thresh;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Make fork plot
if isempty(ax)
    figure;
    ax = gca;
end
heatMap(isnan(heatMap)) = 0;
imagesc(ax, dbins, 0.5*(abins(1:end-1)+abins(2:end)), heatMap)
colormap jet
xlabel('Position on cell depth axis (\mum)')
ylabel('Cell size (\mum^2)')

hold(ax,'on');
plot(-0.5*meanCellWidths, 0.5*(abins(1:end-1)+abins(2:end)), 'w', 'LineWidth', 2);
plot(0.5*meanCellWidths, 0.5*(abins(1:end-1)+abins(2:end)), 'w', 'LineWidth', 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add lines corresponding to cell birth size, division size and initiation
% size

if ~isempty(avgBirthArea)
    plot([dbins(1) dbins(end)], [avgBirthArea, avgBirthArea], 'w--','LineWidth', 1.5);
end

if ~isempty(avgDivisionArea)
    plot([dbins(1) dbins(end)], [avgDivisionArea, avgDivisionArea], 'w--','LineWidth', 1.5);
end

if ~isempty(avgInitiationArea)
    plot([dbins(1) dbins(end)], [avgInitiationArea, avgInitiationArea], 'r--','LineWidth', 1.5);
end

hold(ax,'off');