function tf = has_stats_toolbox()
tf = ~isempty(ver('stats')) && license('test', 'Statistics_Toolbox');
end
