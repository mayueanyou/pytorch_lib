class SelfAttention(nn.Module):
    def __init__(self, dim,linear_projection={'q':True,'k':True,'v':True,'o':True}):
        super().__init__()
        self.scale = dim ** -0.5
        #inner_dim = dim if not increase_dim else dim * heads

        self.key = nn.Linear(dim, dim) if linear_projection['k'] else None
        self.query = nn.Linear(dim, dim) if linear_projection['q'] else None
        self.value = nn.Linear(dim, dim) if linear_projection['v'] else None
        self.out = nn.Linear(dim, dim) if linear_projection['o'] else None

    def forward(self, x, mask=None):
        keys = self.key(x) if self.key is not None else x
        queries = self.query(x) if self.query is not None else x
        values = self.value(x) if self.value is not None else x
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, values)
        out = self.out(out) if self.out is not None else out
        return out

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, dim, heads=1,increase_dim=False):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        inner_dim = dim if not increase_dim else dim * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attention = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out