
import tensorflow as tf
import torch 
import torch.nn as nn
import mobile_cv

def build_layer(x, torch_layer, layer_name=None):
    if isinstance(torch_layer, nn.Conv2d):
        use_bias = torch_layer.bias is not None
        groups = torch_layer.groups 

        if groups == 1:
            zeropad = tf.keras.layers.ZeroPadding2D(padding=torch_layer.padding)
            keras_module = tf.keras.layers.Conv2D(
                filters=torch_layer.out_channels,
                kernel_size=torch_layer.kernel_size,
                strides=torch_layer.stride,
                padding='valid',
                activation=None,
                use_bias=use_bias,
                name=layer_name+"_conv"
            )
            keras_module.build(x.shape)

            weights = [torch_layer.weight.permute((2,3,1,0)).detach().numpy()]
            if use_bias:
                weights.append(torch_layer.bias.detach().numpy())
        else:
            assert groups == torch_layer.out_channels, "Only depth_multiplier=1 supported"

            zeropad = tf.keras.layers.ZeroPadding2D(padding=torch_layer.padding)
            keras_module = tf.keras.layers.DepthwiseConv2D(
                kernel_size=torch_layer.kernel_size,
                strides=torch_layer.stride,
                depth_multiplier=1,
                padding="valid",
                activation=None,
                use_bias=use_bias,
                name=layer_name+"_conv"
            )
            keras_module.build(x.shape)

            weights = [torch_layer.weight.permute((2,3,0,1)).detach().numpy()]
            if use_bias:
                weights.append(torch_layer.bias.detach().numpy())

        keras_module.set_weights(weights)

        return keras_module(zeropad(x))

    elif isinstance(torch_layer, nn.Linear):
        use_bias = torch_layer.bias is not None

        keras_module = tf.keras.layers.Dense(
            filters=torch_layer.out_features,
            activation=None,
            use_bias=use_bias,
            name=layer_name+"_fc"
        )
        keras_module.build(x.shape)

        weights = [torch_layer.weight.detach().numpy()]
        if use_bias:
            weights.append(torch_layer.bias.detach().numpy())

        keras_module.set_weights(weights)
        return keras_module(x)

    elif isinstance(torch_layer, nn.BatchNorm2d):
        keras_module = tf.keras.layers.BatchNormalization(
            momentum=1-torch_layer.momentum,
            epsilon=torch_layer.eps,
            trainable=torch_layer.track_running_stats,
            name=layer_name+"_bn")
        keras_module.build(x.shape)

        weights = [torch_layer.weight.detach().numpy(), 
                    torch_layer.bias.detach().numpy(), 
                    torch_layer.running_mean.detach().numpy(), 
                    torch_layer.running_var.detach().numpy()]
        keras_module.set_weights(weights)
    
        return keras_module(x)

    elif isinstance(torch_layer, nn.ReLU):
        keras_module = tf.keras.layers.ReLU(name=layer_name+"_relu")
        keras_module.build(x.shape)
        return keras_module(x)

    elif isinstance(torch_layer, mobile_cv.arch.fbnet_v2.basic_blocks.HSigmoid):
        relu6 = tf.keras.layers.ReLU(max_value=6, name=layer_name+"_hsigmoid")
        return relu6(x + 3.0) / 6.0

    elif isinstance(torch_layer, nn.Hardswish):
        relu6 = tf.keras.layers.ReLU(max_value=6, name=layer_name+"_hswish")
        return x * relu6(x + 3.0) / 6.0

    elif isinstance(torch_layer, mobile_cv.arch.fbnet_v2.basic_blocks.ConvBNRelu):
        if torch_layer.conv is not None:
            x = build_layer(x, torch_layer.conv, layer_name=layer_name+"_conv")
        if torch_layer.bn is not None:
            x = build_layer(x, torch_layer.bn, layer_name=layer_name+"_bn")
        if torch_layer.relu is not None:
            x = build_layer(x, torch_layer.relu, layer_name=layer_name+"_relu")
        if torch_layer.upsample is not None:
            x = build_layer(x, torch_layer.upsample, layer_name=layer_name+"_upsample")
        return x

    elif isinstance(torch_layer, mobile_cv.arch.fbnet_v2.basic_blocks.SEModule):
        if torch_layer.use_fc:
            raise NotImplementedError

        y = build_layer(x, torch_layer.avg_pool, layer_name=layer_name+"_avgpool")
        y = build_layer(y, torch_layer.se[0], layer_name=layer_name+"_conv1_relu")
        y = build_layer(y, torch_layer.se[1], layer_name=layer_name+"_se")
        y = build_layer(y, torch_layer.se[2], layer_name=layer_name+"_relu")
        y = build_layer([x, y], torch_layer.mul, layer_name=layer_name+"_mul")   
        return y     

    elif isinstance(torch_layer, mobile_cv.arch.fbnet_v2.basic_blocks.TorchAdd):
        return tf.keras.layers.Add(name=layer_name+"_add")(x)

    elif isinstance(torch_layer, mobile_cv.arch.fbnet_v2.basic_blocks.TorchMultiply):
        return tf.keras.layers.Multiply(name=layer_name+"_mult")(x)

    elif isinstance(torch_layer, mobile_cv.arch.fbnet_v2.basic_blocks.Identity):
        out = x
        if torch_layer.conv is not None:
            out = build_layer(x, torch_layer.conv, layer_name=layer_name+"_conv")
        return out

    elif isinstance(torch_layer, mobile_cv.arch.fbnet_v2.irf_block.IRFBlock):
        y = x
        if torch_layer.pw is not None:
            y = build_layer(y, torch_layer.pw, layer_name=layer_name+"_pw")
        if torch_layer.shuffle is not None:
            y = build_layer(y, torch_layer.shuffle, layer_name=layer_name+"_shuffle")
        if torch_layer.upsample is not None:
            y = build_layer(y, torch_layer.upsample, layer_name=layer_name+"_upsample")
        if torch_layer.dw is not None:
            y = build_layer(y, torch_layer.dw, layer_name=layer_name+"_dw")
        if torch_layer.se is not None:
            y = build_layer(y, torch_layer.se, layer_name=layer_name+"_se")
        if torch_layer.pwl is not None:
            y = build_layer(y, torch_layer.pwl, layer_name=layer_name+"_pwl")
        if torch_layer.res_conn is not None:
            y = build_layer([y, x], torch_layer.res_conn, layer_name=layer_name+"_resconn")
        if torch_layer.relu is not None:
            y = build_layer(y, torch_layer.relu, layer_name=layer_name+"_relu")
        return y      

    elif isinstance(torch_layer, mobile_cv.arch.fbnet_v2.irf_block.IRPoolBlock):
        y = x
        if torch_layer.pw is not None:
            y = build_layer(y, torch_layer.pw, layer_name=layer_name+"_pw")
        if torch_layer.pw_se is not None:
            y = build_layer(y, torch_layer.pw_se, layer_name=layer_name+"_pw_se")
        if torch_layer.dw is not None:
            y = build_layer(y, torch_layer.dw, layer_name=layer_name+"_dw")
        if torch_layer.se is not None:
            y = build_layer(y, torch_layer.se, layer_name=layer_name+"_se")
        if torch_layer.pwl is not None:
            y = build_layer(y, torch_layer.pwl, layer_name=layer_name+"_pwl")
        if torch_layer.res_conn is not None:
            y = torch_layer.res_conn.convert_keras(y, x, layer_name=layer_name+"_resconn")
        return y       

    elif isinstance(torch_layer, nn.AdaptiveAvgPool2d):
        keras_module = tf.keras.layers.AveragePooling2D(
            pool_size=(x.shape[1], x.shape[2])
        )
        keras_module.build(x.shape)
        return keras_module(x)

    else:
        raise NotImplementedError

def build_backbone(x, torch_backbone, layer_name=None):
    for name, stage in torch_backbone.stages._modules.items():
        x = build_layer(x, stage, layer_name=name)
    return x

def build_head(x, torch_head, layer_name=None):
    x = build_layer(x, torch_head.avg_pool, layer_name=layer_name+"_avgpool")
    x = build_layer(x, torch_head.conv, layer_name=layer_name+"_conv")
    return x

def build_fbnet(fbnet):
    image_height = image_width = fbnet.arch_def['input_size']
    input_layer = tf.keras.Input(shape=[image_height,image_width,3], batch_size=1, name="input_1")

    x = input_layer
    x = build_backbone(x, fbnet.backbone)
    output_layer = build_head(x, fbnet.head, layer_name="head")
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile()
    return model



        
        



