function Y = channel_shuffle(X, g)
sz = size(X, 1:4);
Y = reshape(X, [sz(1) sz(2) sz(3)/g g sz(4)]);
Y = permute(Y, [1 2 4 3 5]);
Y = reshape(Y, sz);
end
