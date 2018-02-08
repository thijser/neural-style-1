function [vq] = interp_left(x, v, xq)
%INTERP_LEFT Interpolate to the left-nearest point
% x must be sorted.
vq = nan(size(xq));
for ii = 1:length(xq)
  % Find the index in x nearest to xq, only considering smaller x
  [~,jj] = max(x(x<=xq(ii)));
  % Special case, there are no smaller x; extrapolate using [x(1),v(1)]
  if isempty(jj)
    vq(ii) = v(1);
  else
    vq(ii) = v(jj);
  end % if
end % for
end % function