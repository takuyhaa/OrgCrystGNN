import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    MetaLayer,
    GCNConv,
    CGConv,
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter
from torch_geometric.nn.models.schnet import InteractionBlock
from sklearn.ensemble import RandomForestRegressor

# Megnet
class Megnet_EdgeModel(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(Megnet_EdgeModel, self).__init__()
        self.act=act
        self.fc_layers = fc_layers
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
                
        self.edge_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim * 4, dim)
                self.edge_mlp.append(lin)       
            else:      
                lin = torch.nn.Linear(dim, dim)
                self.edge_mlp.append(lin) 
            if self.batch_norm == "True":
                bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)            

    def forward(self, src, dest, edge_attr, u, batch):
        comb = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        for i in range(0, len(self.edge_mlp)):
            if i == 0:
                out = self.edge_mlp[i](comb)
                out = getattr(F, self.act)(out)  
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)
            else:    
                out = self.edge_mlp[i](out)
                out = getattr(F, self.act)(out)    
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)   
        return out


class Megnet_NodeModel(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(Megnet_NodeModel, self).__init__()
        self.act=act
        self.fc_layers = fc_layers
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
                
        self.node_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim * 3, dim)
                self.node_mlp.append(lin)       
            else:      
                lin = torch.nn.Linear(dim, dim)
                self.node_mlp.append(lin) 
            if self.batch_norm == "True":
                bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)    

    def forward(self, x, edge_index, edge_attr, u, batch):
        # row, col = edge_index
        v_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        comb = torch.cat([x, v_e, u[batch]], dim=1)
        for i in range(0, len(self.node_mlp)):
            if i == 0:
                out = self.node_mlp[i](comb)
                out = getattr(F, self.act)(out)  
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)
            else:    
                out = self.node_mlp[i](out)
                out = getattr(F, self.act)(out)    
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)   
        return out


class Megnet_GlobalModel(torch.nn.Module):
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(Megnet_GlobalModel, self).__init__()
        self.act=act
        self.fc_layers = fc_layers
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
                
        self.global_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim * 3, dim)
                self.global_mlp.append(lin)       
            else:      
                lin = torch.nn.Linear(dim, dim)
                self.global_mlp.append(lin) 
            if self.batch_norm == "True":
                bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)  

    def forward(self, x, edge_index, edge_attr, u, batch):
        u_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        u_e = scatter_mean(u_e, batch, dim=0)
        u_v = scatter_mean(x, batch, dim=0)
        comb = torch.cat([u_e, u_v, u], dim=1)
        for i in range(0, len(self.global_mlp)):
            if i == 0:
                out = self.global_mlp[i](comb)
                out = getattr(F, self.act)(out)  
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)
            else:    
                out = self.global_mlp[i](out)
                out = getattr(F, self.act)(out)    
                if self.batch_norm == "True":
                    out = self.bn_list[i](out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)   
        return out


class MEGNet(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        dim3=64,
        pre_fc_count=1,
        gc_count=3,
        gc_fc_count=2,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(MEGNet, self).__init__()
        
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.pool = pool
        if pool == "global_mean_pool":
            self.pool_reduce="mean"
        elif pool== "global_max_pool":
            self.pool_reduce="max" 
        elif pool== "global_add_pool":
            self.pool_reduce="add"             
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        self.gradcam = False
        self.manifold = False
        
        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            gc_dim = data.num_features
        else:
            gc_dim = dim1
        ##Determine post_fc dimension
        post_fc_dim = dim3
        ##Determine output dimension length
        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])

        ##Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(data.num_features, dim1)
                    self.pre_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()

        ##Set up GNN layers
        self.e_embed_list = torch.nn.ModuleList()
        self.x_embed_list = torch.nn.ModuleList()
        self.u_embed_list = torch.nn.ModuleList()   
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            if i == 0:
                e_embed = Sequential(
                    Linear(data.num_edge_features, dim3), ReLU(), Linear(dim3, dim3), ReLU()
                )
                x_embed = Sequential(
                    Linear(gc_dim, dim3), ReLU(), Linear(dim3, dim3), ReLU()
                )
                u_embed = Sequential(
                    Linear((data[0].u.shape[1]), dim3), ReLU(), Linear(dim3, dim3), ReLU()
                )
                self.e_embed_list.append(e_embed)
                self.x_embed_list.append(x_embed)
                self.u_embed_list.append(u_embed)
                self.conv_list.append(
                    MetaLayer(
                        Megnet_EdgeModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_NodeModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_GlobalModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                    )
                )
            elif i > 0:
                e_embed = Sequential(Linear(dim3, dim3), ReLU(), Linear(dim3, dim3), ReLU())
                x_embed = Sequential(Linear(dim3, dim3), ReLU(), Linear(dim3, dim3), ReLU())
                u_embed = Sequential(Linear(dim3, dim3), ReLU(), Linear(dim3, dim3), ReLU())
                self.e_embed_list.append(e_embed)
                self.x_embed_list.append(x_embed)
                self.u_embed_list.append(u_embed)
                self.conv_list.append(
                    MetaLayer(
                        Megnet_EdgeModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_NodeModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_GlobalModel(dim3, self.act, self.batch_norm, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                    )
                )

        ##Set up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)
        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 5, dim2)
                    elif self.pool_order == "early" and self.pool != "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 3, dim2)
                    elif self.pool_order == "late":
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim * 5, output_dim)
            elif self.pool_order == "early" and self.pool != "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim * 3, output_dim)              
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)   

        ##Set up set2set pooling (if used)
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set_x = Set2Set(post_fc_dim, processing_steps=3)
            self.set2set_e = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set_x = Set2Set(output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)
        # placeholder for the gradients
        self.gradients = None
    
    
        # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad


    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    

    # method for the activation exctraction
    def get_activations(self, data):
        # Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        # GNN layers
        for i in range(0, len(self.conv_list)):
            if i == 0:
                if len(self.pre_lin_list) == 0:
                    e_temp = self.e_embed_list[i](data.edge_attr)
                    x_temp = self.x_embed_list[i](data.x)
                    u_temp = self.u_embed_list[i](data.u)
                    x_out, e_out, u_out = self.conv_list[i](
                        x_temp, data.edge_index, e_temp, u_temp, data.batch
                    )
                    x = torch.add(x_out, x_temp)
                    e = torch.add(e_out, e_temp)
                    u = torch.add(u_out, u_temp)
                else:
                    e_temp = self.e_embed_list[i](data.edge_attr)
                    x_temp = self.x_embed_list[i](out)
                    u_temp = self.u_embed_list[i](data.u)
                    x_out, e_out, u_out = self.conv_list[i](
                        x_temp, data.edge_index, e_temp, u_temp, data.batch
                    )
                    x = torch.add(x_out, x_temp)
                    e = torch.add(e_out, e_temp)
                    u = torch.add(u_out, u_temp)

            elif i > 0:
                e_temp = self.e_embed_list[i](e)
                x_temp = self.x_embed_list[i](x)
                u_temp = self.u_embed_list[i](u)
                x_out, e_out, u_out = self.conv_list[i](
                    x_temp, data.edge_index, e_temp, u_temp, data.batch
                )
                x = torch.add(x_out, x)
                e = torch.add(e_out, e)
                u = torch.add(u_out, u)
        return x
    

    def forward(self, data):

        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        ##GNN layers        
        for i in range(0, len(self.conv_list)):
            if i == 0:
                if len(self.pre_lin_list) == 0:
                    e_temp = self.e_embed_list[i](data.edge_attr)
                    x_temp = self.x_embed_list[i](data.x)
                    u_temp = self.u_embed_list[i](data.u)
                    x_out, e_out, u_out = self.conv_list[i](
                        x_temp, data.edge_index, e_temp, u_temp, data.batch
                    )
                    x = torch.add(x_out, x_temp)
                    e = torch.add(e_out, e_temp)
                    u = torch.add(u_out, u_temp)
                else:
                    e_temp = self.e_embed_list[i](data.edge_attr)
                    x_temp = self.x_embed_list[i](out)
                    u_temp = self.u_embed_list[i](data.u)
                    x_out, e_out, u_out = self.conv_list[i](
                        x_temp, data.edge_index, e_temp, u_temp, data.batch
                    )
                    x = torch.add(x_out, x_temp)
                    e = torch.add(e_out, e_temp)
                    u = torch.add(u_out, u_temp)
                    
            elif i > 0:
                e_temp = self.e_embed_list[i](e)
                x_temp = self.x_embed_list[i](x)
                u_temp = self.u_embed_list[i](u)
                x_out, e_out, u_out = self.conv_list[i](
                    x_temp, data.edge_index, e_temp, u_temp, data.batch
                )
                x = torch.add(x_out, x)
                e = torch.add(e_out, e)
                u = torch.add(u_out, u)

                if self.gradcam is True and i == len(self.conv_list) - 1:
                    x.register_hook(self.activations_hook)

        ##Post-GNN dense layers
        if self.pool_order == "early":
            if self.pool == "set2set":
                x_pool = self.set2set_x(x, data.batch)
                e = scatter(e, data.edge_index[0, :], dim=0, reduce="mean")
                e_pool = self.set2set_e(e, data.batch)
                out = torch.cat([x_pool, e_pool, u], dim=1)                
            else:
                x_pool = scatter(x, data.batch, dim=0, reduce=self.pool_reduce)
                e_pool = scatter(e, data.edge_index[0, :], dim=0, reduce=self.pool_reduce)
                e_pool = scatter(e_pool, data.batch, dim=0, reduce=self.pool_reduce)
                out = torch.cat([x_pool, e_pool, u], dim=1)
            if self.manifold is True:
                self.feat_manifold = out
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)

        ##currently only uses node features for late pooling
        elif self.pool_order == "late":
            out = x
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)
            if self.pool == "set2set":
                out = self.set2set_x(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out

        


class GCN(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        dim3=None,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(GCN, self).__init__()
        
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        
        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"
        if pre_fc_count == 0:
            gc_dim = data.num_features
        else:
            gc_dim = dim1
        ##Determine post_fc dimension
        if pre_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1
        ##Determine output dimension length
        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])

        ##Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(data.num_features, dim1)
                    self.pre_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()

        ##Set up GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            conv = GCNConv(
                gc_dim, gc_dim, improved=True, add_self_loops=False
            )
            self.conv_list.append(conv)
            ##Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm == "True":
                bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        ##Set up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)
        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                    else:
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim*2, output_dim)
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)   

        ##Set up set2set pooling (if used)
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)

    def forward(self, data):

        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        ##GNN layers
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    out = self.conv_list[i](data.x, data.edge_index, data.edge_weight)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](data.x, data.edge_index, data.edge_weight)
            else:
                if self.batch_norm == "True":
                    out = self.conv_list[i](out, data.edge_index, data.edge_weight)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](out, data.edge_index, data.edge_weight)            
            out = getattr(F, self.act)(out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        ##Post-GNN dense layers
        if self.pool_order == "early":
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)

        elif self.pool_order == "late":
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out
        
        
# CGCNN
class CGCNN(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        dim3=None,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(CGCNN, self).__init__()
        
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        
        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"        
        if pre_fc_count == 0:
            gc_dim = data.num_features
        else:
            gc_dim = dim1
        ##Determine post_fc dimension
        if pre_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1
        ##Determine output dimension length
        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])

        ##Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(data.num_features, dim1)
                    self.pre_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()

        ##Set up GNN layers
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            conv = CGConv(
                gc_dim, data.num_edge_features, aggr="mean", batch_norm=False
            )
            self.conv_list.append(conv)
            ##Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm == "True":
                bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        ##Set up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)
        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                    else:
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim*2, output_dim)
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)   

        ##Set up set2set pooling (if used)
        ##Should processing_setps be a hypereparameter?
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)

    def forward(self, data):

        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        ##GNN layers
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    out = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](data.x, data.edge_index, data.edge_attr)
            else:
                if self.batch_norm == "True":
                    out = self.conv_list[i](out, data.edge_index, data.edge_attr)
                    out = self.bn_list[i](out)
                else:
                    out = self.conv_list[i](out, data.edge_index, data.edge_attr)            
            #out = getattr(F, self.act)(out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        ##Post-GNN dense layers
        if self.pool_order == "early":
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)

        elif self.pool_order == "late":
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out

        
        
# Schnet
class SchNet(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        dim3=64,
        cutoff=8,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="relu",
        dropout_rate=0.0,
        **kwargs
    ):
        super(SchNet, self).__init__()
        
        if batch_track_stats == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 
        self.batch_norm = batch_norm
        self.pool = pool
        self.act = act
        self.pool_order = pool_order
        self.dropout_rate = dropout_rate
        
        ##Determine gc dimension dimension
        assert gc_count > 0, "Need at least 1 GC layer"        
        if pre_fc_count == 0:
            gc_dim = data.num_features
        else:
            gc_dim = dim1
        ##Determine post_fc dimension
        if pre_fc_count == 0:
            post_fc_dim = data.num_features
        else:
            post_fc_dim = dim1
        ##Determine output dimension length
        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])

        ##Set up pre-GNN dense layers (NOTE: in v0.1 this is always set to 1 layer)
        if pre_fc_count > 0:
            self.pre_lin_list = torch.nn.ModuleList()
            for i in range(pre_fc_count):
                if i == 0:
                    lin = torch.nn.Linear(data.num_features, dim1)
                    self.pre_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim1, dim1)
                    self.pre_lin_list.append(lin)
        elif pre_fc_count == 0:
            self.pre_lin_list = torch.nn.ModuleList()

        ##Set up GNN layers       
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            conv = InteractionBlock(gc_dim, data.num_edge_features, dim3, cutoff)
            self.conv_list.append(conv)
            ##Track running stats set to false can prevent some instabilities; this causes other issues with different val/test performance from loader size?
            if self.batch_norm == "True":
                bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
                self.bn_list.append(bn)

        ##Set up post-GNN dense layers (NOTE: in v0.1 there was a minimum of 2 dense layers, and fc_count(now post_fc_count) added to this number. In the current version, the minimum is zero)
        if post_fc_count > 0:
            self.post_lin_list = torch.nn.ModuleList()
            for i in range(post_fc_count):
                if i == 0:
                    ##Set2set pooling has doubled dimension
                    if self.pool_order == "early" and self.pool == "set2set":
                        lin = torch.nn.Linear(post_fc_dim * 2, dim2)
                    else:
                        lin = torch.nn.Linear(post_fc_dim, dim2)
                    self.post_lin_list.append(lin)
                else:
                    lin = torch.nn.Linear(dim2, dim2)
                    self.post_lin_list.append(lin)
            self.lin_out = torch.nn.Linear(dim2, output_dim)

        elif post_fc_count == 0:
            self.post_lin_list = torch.nn.ModuleList()
            if self.pool_order == "early" and self.pool == "set2set":
                self.lin_out = torch.nn.Linear(post_fc_dim*2, output_dim)
            else:
                self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)   

        ##Set up set2set pooling (if used)
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)
            # workaround for doubled dimension by set2set; if late pooling not reccomended to use set2set
            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)

    def forward(self, data):

        ##Pre-GNN dense layers
        for i in range(0, len(self.pre_lin_list)):
            if i == 0:
                out = self.pre_lin_list[i](data.x)
                out = getattr(F, self.act)(out)
            else:
                out = self.pre_lin_list[i](out)
                out = getattr(F, self.act)(out)

        ##GNN layers
        for i in range(0, len(self.conv_list)):
            if len(self.pre_lin_list) == 0 and i == 0:
                if self.batch_norm == "True":
                    out = data.x + self.conv_list[i](data.x, data.edge_index, data.edge_weight, data.edge_attr)
                    out = self.bn_list[i](out)
                else:
                    out = data.x + self.conv_list[i](data.x, data.edge_index, data.edge_weight, data.edge_attr)
            else:
                if self.batch_norm == "True":
                    out = out + self.conv_list[i](out, data.edge_index, data.edge_weight, data.edge_attr)
                    out = self.bn_list[i](out)
                else:
                    out = out + self.conv_list[i](out, data.edge_index, data.edge_weight, data.edge_attr)            
            #out = getattr(F, self.act)(out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        ##Post-GNN dense layers
        if self.pool_order == "early":
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)

        elif self.pool_order == "late":
            for i in range(0, len(self.post_lin_list)):
                out = self.post_lin_list[i](out)
                out = getattr(F, self.act)(out)
            out = self.lin_out(out)
            if self.pool == "set2set":
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)
            else:
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
                
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out
        

class NN(torch.nn.Module):
    def __init__(self, data, n_dim, n_layers, **kwargs):
        super(NN, self).__init__()
        
        self.lin1 = torch.nn.Linear(data.shape[1], n_dim)
        self.lin_list = torch.nn.ModuleList(
            [torch.nn.Linear(n_dim, n_dim) for i in range(n_layers)]
        )
        self.lin2 = torch.nn.Linear(n_dim, 1)

    def forward(self, data):
        out = F.relu(self.lin1(data))
        for layer in self.lin_list:
            out = F.relu(layer(out))
        out = self.lin2(out)
        if out.shape[1] == 1:
            return out.view(-1)
        else:
            return out


def RF(n_estimators, max_features, min_samples_leaf, max_depth):
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_features=max_features,
                                  min_samples_leaf=min_samples_leaf,
                                  max_depth=max_depth)
    return model


def makemodel(model_name, **model_params):
    if model_name == 'GCN':
        model = GCN(**model_params)
    elif model_name == 'MEGNet':
        model = MEGNet(**model_params)
    elif model_name == 'SchNet':
        model = SchNet(**model_params)
    elif model_name == 'CGCNN':
        model = CGCNN(**model_params)
    elif model_name == 'NN':
        model = NN(**model_params)
    return model