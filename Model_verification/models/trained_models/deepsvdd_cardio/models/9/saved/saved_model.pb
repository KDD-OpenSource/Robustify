
Ö
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
7
Square
x"T
y"T"
Ttype:
2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Î 
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:*
dtype0
~
net_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namenet_output/kernel
w
%net_output/kernel/Read/ReadVariableOpReadVariableOpnet_output/kernel*
_output_shapes

:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_19/kernel/m

*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes

:*
dtype0

Adam/net_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/net_output/kernel/m

,Adam/net_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/net_output/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_19/kernel/v

*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes

:*
dtype0

Adam/net_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/net_output/kernel/v

,Adam/net_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/net_output/kernel/v*
_output_shapes

:*
dtype0

NoOpNoOp
¦ 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*á
value×BÔ BÍ
ÿ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 


kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*


kernel
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses*

"	keras_api* 

#	keras_api* 

$	keras_api* 

%	keras_api* 

&	keras_api* 

'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
h
-iter

.beta_1

/beta_2
	0decay
1learning_ratemPmQvRvS*
* 

0
1*

0
1*
* 
°
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

7serving_default* 
_Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
°
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
=activity_regularizer_fn
*&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUEnet_output/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
°
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
Dactivity_regularizer_fn
*!&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
2
3
4
5
6
7
	8*

K0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Ltotal
	Mcount
N	variables
O	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

N	variables*
|
VARIABLE_VALUEAdam/dense_19/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/net_output/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_19/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/net_output/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
serving_default_input_20Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
×
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_20dense_19/kernelnet_output/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_104526
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_19/kernel/Read/ReadVariableOp%net_output/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp,Adam/net_output/kernel/m/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp,Adam/net_output/kernel/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_104647
þ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_19/kernelnet_output/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_19/kernel/mAdam/net_output/kernel/mAdam/dense_19/kernel/vAdam/net_output/kernel/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_104696Ú

­
D__inference_dense_19_layer_call_and_return_conditional_losses_104577

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý4
£
D__inference_model_49_layer_call_and_return_conditional_losses_104283

inputs!
dense_19_104245:#
net_output_104256:
identity

identity_1

identity_2

identity_3¢ dense_19/StatefulPartitionedCall¢"net_output/StatefulPartitionedCallÝ
 dense_19/StatefulPartitionedCallStatefulPartitionedCallinputsdense_19_104245*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_104121Ê
,dense_19/ActivityRegularizer/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *9
f4R2
0__inference_dense_19_activity_regularizer_104093{
"dense_19/ActivityRegularizer/ShapeShape)dense_19/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ®
$dense_19/ActivityRegularizer/truedivRealDiv5dense_19/ActivityRegularizer/PartitionedCall:output:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
"net_output/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0net_output_104256*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_net_output_layer_call_and_return_conditional_losses_104141Ð
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *;
f6R4
2__inference_net_output_activity_regularizer_104106
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ´
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_19/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<D?uÿ>ÍÌÌ=^?ÍÌÌ=B7i>àj>ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÖC>ÍÌÌ=ÍÌÌ=ÍÌÌ=¡
tf.math.subtract_19/SubSub+net_output/StatefulPartitionedCall:output:0"tf.math.subtract_19/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_19/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_19/PowPowtf.math.subtract_19/Sub:z:0tf.math.pow_19/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_19/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_19/SumSumtf.math.pow_19/Pow:z:04tf.math.reduce_sum_19/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_19/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_19/MeanMean"tf.math.reduce_sum_19/Sum:output:0%tf.math.reduce_mean_19/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *ê7
tf.__operators__.add_19/AddV2AddV2$tf.math.reduce_mean_19/Mean:output:0"tf.__operators__.add_19/y:output:0*
T0*
_output_shapes
: Ë
add_loss_19/PartitionedCallPartitionedCall!tf.__operators__.add_19/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_add_loss_19_layer_call_and_return_conditional_losses_104168m
IdentityIdentity"tf.math.reduce_sum_19/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: d

Identity_3Identity$add_loss_19/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^dense_19/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
5
¥
D__inference_model_49_layer_call_and_return_conditional_losses_104346
input_20!
dense_19_104308:#
net_output_104319:
identity

identity_1

identity_2

identity_3¢ dense_19/StatefulPartitionedCall¢"net_output/StatefulPartitionedCallß
 dense_19/StatefulPartitionedCallStatefulPartitionedCallinput_20dense_19_104308*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_104121Ê
,dense_19/ActivityRegularizer/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *9
f4R2
0__inference_dense_19_activity_regularizer_104093{
"dense_19/ActivityRegularizer/ShapeShape)dense_19/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ®
$dense_19/ActivityRegularizer/truedivRealDiv5dense_19/ActivityRegularizer/PartitionedCall:output:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
"net_output/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0net_output_104319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_net_output_layer_call_and_return_conditional_losses_104141Ð
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *;
f6R4
2__inference_net_output_activity_regularizer_104106
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ´
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_19/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<D?uÿ>ÍÌÌ=^?ÍÌÌ=B7i>àj>ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÖC>ÍÌÌ=ÍÌÌ=ÍÌÌ=¡
tf.math.subtract_19/SubSub+net_output/StatefulPartitionedCall:output:0"tf.math.subtract_19/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_19/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_19/PowPowtf.math.subtract_19/Sub:z:0tf.math.pow_19/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_19/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_19/SumSumtf.math.pow_19/Pow:z:04tf.math.reduce_sum_19/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_19/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_19/MeanMean"tf.math.reduce_sum_19/Sum:output:0%tf.math.reduce_mean_19/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *ê7
tf.__operators__.add_19/AddV2AddV2$tf.math.reduce_mean_19/Mean:output:0"tf.__operators__.add_19/y:output:0*
T0*
_output_shapes
: Ë
add_loss_19/PartitionedCallPartitionedCall!tf.__operators__.add_19/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_add_loss_19_layer_call_and_return_conditional_losses_104168m
IdentityIdentity"tf.math.reduce_sum_19/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: d

Identity_3Identity$add_loss_19/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^dense_19/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_20

¯
F__inference_net_output_layer_call_and_return_conditional_losses_104141

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

­
D__inference_dense_19_layer_call_and_return_conditional_losses_104121

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

}
)__inference_dense_19_layer_call_fn_104533

inputs
unknown:
identity¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_104121o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

I
2__inference_net_output_activity_regularizer_104106
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
ó$
Ú
__inference__traced_save_104647
file_prefix.
*savev2_dense_19_kernel_read_readvariableop0
,savev2_net_output_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_19_kernel_m_read_readvariableop7
3savev2_adam_net_output_kernel_m_read_readvariableop5
1savev2_adam_dense_19_kernel_v_read_readvariableop7
3savev2_adam_net_output_kernel_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ä
valueºB·B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B ÷
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_19_kernel_read_readvariableop,savev2_net_output_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_19_kernel_m_read_readvariableop3savev2_adam_net_output_kernel_m_read_readvariableop1savev2_adam_dense_19_kernel_v_read_readvariableop3savev2_adam_net_output_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*a
_input_shapesP
N: ::: : : : : : : ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: 
ý4
£
D__inference_model_49_layer_call_and_return_conditional_losses_104175

inputs!
dense_19_104122:#
net_output_104142:
identity

identity_1

identity_2

identity_3¢ dense_19/StatefulPartitionedCall¢"net_output/StatefulPartitionedCallÝ
 dense_19/StatefulPartitionedCallStatefulPartitionedCallinputsdense_19_104122*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_104121Ê
,dense_19/ActivityRegularizer/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *9
f4R2
0__inference_dense_19_activity_regularizer_104093{
"dense_19/ActivityRegularizer/ShapeShape)dense_19/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ®
$dense_19/ActivityRegularizer/truedivRealDiv5dense_19/ActivityRegularizer/PartitionedCall:output:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
"net_output/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0net_output_104142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_net_output_layer_call_and_return_conditional_losses_104141Ð
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *;
f6R4
2__inference_net_output_activity_regularizer_104106
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ´
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_19/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<D?uÿ>ÍÌÌ=^?ÍÌÌ=B7i>àj>ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÖC>ÍÌÌ=ÍÌÌ=ÍÌÌ=¡
tf.math.subtract_19/SubSub+net_output/StatefulPartitionedCall:output:0"tf.math.subtract_19/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_19/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_19/PowPowtf.math.subtract_19/Sub:z:0tf.math.pow_19/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_19/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_19/SumSumtf.math.pow_19/Pow:z:04tf.math.reduce_sum_19/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_19/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_19/MeanMean"tf.math.reduce_sum_19/Sum:output:0%tf.math.reduce_mean_19/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *ê7
tf.__operators__.add_19/AddV2AddV2$tf.math.reduce_mean_19/Mean:output:0"tf.__operators__.add_19/y:output:0*
T0*
_output_shapes
: Ë
add_loss_19/PartitionedCallPartitionedCall!tf.__operators__.add_19/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_add_loss_19_layer_call_and_return_conditional_losses_104168m
IdentityIdentity"tf.math.reduce_sum_19/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: d

Identity_3Identity$add_loss_19/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^dense_19/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

G
0__inference_dense_19_activity_regularizer_104093
x
identity6
SquareSquarex*
T0*
_output_shapes
:9
RankRank
Square:y:0*
T0*
_output_shapes
: M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
ó

®
J__inference_net_output_layer_call_and_return_all_conditional_losses_104558

inputs
unknown:
identity

identity_1¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_net_output_layer_call_and_return_conditional_losses_104141¦
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *;
f6R4
2__inference_net_output_activity_regularizer_104106o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª:
¢
!__inference__wrapped_model_104080
input_20B
0model_49_dense_19_matmul_readvariableop_resource:D
2model_49_net_output_matmul_readvariableop_resource:
identity¢'model_49/dense_19/MatMul/ReadVariableOp¢)model_49/net_output/MatMul/ReadVariableOp
'model_49/dense_19/MatMul/ReadVariableOpReadVariableOp0model_49_dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
model_49/dense_19/MatMulMatMulinput_20/model_49/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
model_49/dense_19/ReluRelu"model_49/dense_19/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,model_49/dense_19/ActivityRegularizer/SquareSquare$model_49/dense_19/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
+model_49/dense_19/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¹
)model_49/dense_19/ActivityRegularizer/SumSum0model_49/dense_19/ActivityRegularizer/Square:y:04model_49/dense_19/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: p
+model_49/dense_19/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    »
)model_49/dense_19/ActivityRegularizer/mulMul4model_49/dense_19/ActivityRegularizer/mul/x:output:02model_49/dense_19/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
+model_49/dense_19/ActivityRegularizer/ShapeShape$model_49/dense_19/Relu:activations:0*
T0*
_output_shapes
:
9model_49/dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
;model_49/dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
;model_49/dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3model_49/dense_19/ActivityRegularizer/strided_sliceStridedSlice4model_49/dense_19/ActivityRegularizer/Shape:output:0Bmodel_49/dense_19/ActivityRegularizer/strided_slice/stack:output:0Dmodel_49/dense_19/ActivityRegularizer/strided_slice/stack_1:output:0Dmodel_49/dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask 
*model_49/dense_19/ActivityRegularizer/CastCast<model_49/dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¸
-model_49/dense_19/ActivityRegularizer/truedivRealDiv-model_49/dense_19/ActivityRegularizer/mul:z:0.model_49/dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
)model_49/net_output/MatMul/ReadVariableOpReadVariableOp2model_49_net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¯
model_49/net_output/MatMulMatMul$model_49/dense_19/Relu:activations:01model_49/net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
model_49/net_output/ReluRelu$model_49/net_output/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.model_49/net_output/ActivityRegularizer/SquareSquare&model_49/net_output/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
-model_49/net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¿
+model_49/net_output/ActivityRegularizer/SumSum2model_49/net_output/ActivityRegularizer/Square:y:06model_49/net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: r
-model_49/net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Á
+model_49/net_output/ActivityRegularizer/mulMul6model_49/net_output/ActivityRegularizer/mul/x:output:04model_49/net_output/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 
-model_49/net_output/ActivityRegularizer/ShapeShape&model_49/net_output/Relu:activations:0*
T0*
_output_shapes
:
;model_49/net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=model_49/net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=model_49/net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5model_49/net_output/ActivityRegularizer/strided_sliceStridedSlice6model_49/net_output/ActivityRegularizer/Shape:output:0Dmodel_49/net_output/ActivityRegularizer/strided_slice/stack:output:0Fmodel_49/net_output/ActivityRegularizer/strided_slice/stack_1:output:0Fmodel_49/net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¤
,model_49/net_output/ActivityRegularizer/CastCast>model_49/net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ¾
/model_49/net_output/ActivityRegularizer/truedivRealDiv/model_49/net_output/ActivityRegularizer/mul:z:00model_49/net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: §
"model_49/tf.math.subtract_19/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<D?uÿ>ÍÌÌ=^?ÍÌÌ=B7i>àj>ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÖC>ÍÌÌ=ÍÌÌ=ÍÌÌ=®
 model_49/tf.math.subtract_19/SubSub&model_49/net_output/Relu:activations:0+model_49/tf.math.subtract_19/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
model_49/tf.math.pow_19/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¢
model_49/tf.math.pow_19/PowPow$model_49/tf.math.subtract_19/Sub:z:0&model_49/tf.math.pow_19/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4model_49/tf.math.reduce_sum_19/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ·
"model_49/tf.math.reduce_sum_19/SumSummodel_49/tf.math.pow_19/Pow:z:0=model_49/tf.math.reduce_sum_19/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
%model_49/tf.math.reduce_mean_19/ConstConst*
_output_shapes
:*
dtype0*
valueB: ª
$model_49/tf.math.reduce_mean_19/MeanMean+model_49/tf.math.reduce_sum_19/Sum:output:0.model_49/tf.math.reduce_mean_19/Const:output:0*
T0*
_output_shapes
: g
"model_49/tf.__operators__.add_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *ê7¬
&model_49/tf.__operators__.add_19/AddV2AddV2-model_49/tf.math.reduce_mean_19/Mean:output:0+model_49/tf.__operators__.add_19/y:output:0*
T0*
_output_shapes
: v
IdentityIdentity+model_49/tf.math.reduce_sum_19/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp(^model_49/dense_19/MatMul/ReadVariableOp*^model_49/net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2R
'model_49/dense_19/MatMul/ReadVariableOp'model_49/dense_19/MatMul/ReadVariableOp2V
)model_49/net_output/MatMul/ReadVariableOp)model_49/net_output/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_20
¸
s
G__inference_add_loss_19_layer_call_and_return_conditional_losses_104168

inputs
identity

identity_1=
IdentityIdentityinputs*
T0*
_output_shapes
: ?

Identity_1Identityinputs*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs


+__inference_net_output_layer_call_fn_104549

inputs
unknown:
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_net_output_layer_call_and_return_conditional_losses_104141o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â7
Ï
D__inference_model_49_layer_call_and_return_conditional_losses_104466

inputs9
'dense_19_matmul_readvariableop_resource:;
)net_output_matmul_readvariableop_resource:
identity

identity_1

identity_2

identity_3¢dense_19/MatMul/ReadVariableOp¢ net_output/MatMul/ReadVariableOp
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype0{
dense_19/MatMulMatMulinputs&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_19/ReluReludense_19/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
#dense_19/ActivityRegularizer/SquareSquaredense_19/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
"dense_19/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_19/ActivityRegularizer/SumSum'dense_19/ActivityRegularizer/Square:y:0+dense_19/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_19/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *     
 dense_19/ActivityRegularizer/mulMul+dense_19/ActivityRegularizer/mul/x:output:0)dense_19/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: m
"dense_19/ActivityRegularizer/ShapeShapedense_19/Relu:activations:0*
T0*
_output_shapes
:z
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
$dense_19/ActivityRegularizer/truedivRealDiv$dense_19/ActivityRegularizer/mul:z:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
 net_output/MatMul/ReadVariableOpReadVariableOp)net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
net_output/MatMulMatMuldense_19/Relu:activations:0(net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
net_output/ReluRelunet_output/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%net_output/ActivityRegularizer/SquareSquarenet_output/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"net_output/ActivityRegularizer/SumSum)net_output/ActivityRegularizer/Square:y:0-net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: i
$net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
"net_output/ActivityRegularizer/mulMul-net_output/ActivityRegularizer/mul/x:output:0+net_output/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: q
$net_output/ActivityRegularizer/ShapeShapenet_output/Relu:activations:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: £
&net_output/ActivityRegularizer/truedivRealDiv&net_output/ActivityRegularizer/mul:z:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_19/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<D?uÿ>ÍÌÌ=^?ÍÌÌ=B7i>àj>ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÖC>ÍÌÌ=ÍÌÌ=ÍÌÌ=
tf.math.subtract_19/SubSubnet_output/Relu:activations:0"tf.math.subtract_19/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_19/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_19/PowPowtf.math.subtract_19/Sub:z:0tf.math.pow_19/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_19/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_19/SumSumtf.math.pow_19/Pow:z:04tf.math.reduce_sum_19/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_19/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_19/MeanMean"tf.math.reduce_sum_19/Sum:output:0%tf.math.reduce_mean_19/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *ê7
tf.__operators__.add_19/AddV2AddV2$tf.math.reduce_mean_19/Mean:output:0"tf.__operators__.add_19/y:output:0*
T0*
_output_shapes
: m
IdentityIdentity"tf.math.reduce_sum_19/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: a

Identity_3Identity!tf.__operators__.add_19/AddV2:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^dense_19/MatMul/ReadVariableOp!^net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2D
 net_output/MatMul/ReadVariableOp net_output/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç

)__inference_model_49_layer_call_fn_104417

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_49_layer_call_and_return_conditional_losses_104283k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î6
à
"__inference__traced_restore_104696
file_prefix2
 assignvariableop_dense_19_kernel:6
$assignvariableop_1_net_output_kernel:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: ;
)assignvariableop_9_adam_dense_19_kernel_m:>
,assignvariableop_10_adam_net_output_kernel_m:<
*assignvariableop_11_adam_dense_19_kernel_v:>
,assignvariableop_12_adam_net_output_kernel_v:
identity_14¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ä
valueºB·B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B ä
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_19_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_net_output_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp)assignvariableop_9_adam_dense_19_kernel_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp,assignvariableop_10_adam_net_output_kernel_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp*assignvariableop_11_adam_dense_19_kernel_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp,assignvariableop_12_adam_net_output_kernel_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 í
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_14IdentityIdentity_13:output:0^NoOp_1*
T0*
_output_shapes
: Ú
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_14Identity_14:output:0*/
_input_shapes
: : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ç

)__inference_model_49_layer_call_fn_104405

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_49_layer_call_and_return_conditional_losses_104175k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
5
¥
D__inference_model_49_layer_call_and_return_conditional_losses_104387
input_20!
dense_19_104349:#
net_output_104360:
identity

identity_1

identity_2

identity_3¢ dense_19/StatefulPartitionedCall¢"net_output/StatefulPartitionedCallß
 dense_19/StatefulPartitionedCallStatefulPartitionedCallinput_20dense_19_104349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_104121Ê
,dense_19/ActivityRegularizer/PartitionedCallPartitionedCall)dense_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *9
f4R2
0__inference_dense_19_activity_regularizer_104093{
"dense_19/ActivityRegularizer/ShapeShape)dense_19/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:z
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ®
$dense_19/ActivityRegularizer/truedivRealDiv5dense_19/ActivityRegularizer/PartitionedCall:output:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
"net_output/StatefulPartitionedCallStatefulPartitionedCall)dense_19/StatefulPartitionedCall:output:0net_output_104360*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_net_output_layer_call_and_return_conditional_losses_104141Ð
.net_output/ActivityRegularizer/PartitionedCallPartitionedCall+net_output/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *;
f6R4
2__inference_net_output_activity_regularizer_104106
$net_output/ActivityRegularizer/ShapeShape+net_output/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: ´
&net_output/ActivityRegularizer/truedivRealDiv7net_output/ActivityRegularizer/PartitionedCall:output:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_19/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<D?uÿ>ÍÌÌ=^?ÍÌÌ=B7i>àj>ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÖC>ÍÌÌ=ÍÌÌ=ÍÌÌ=¡
tf.math.subtract_19/SubSub+net_output/StatefulPartitionedCall:output:0"tf.math.subtract_19/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_19/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_19/PowPowtf.math.subtract_19/Sub:z:0tf.math.pow_19/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_19/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_19/SumSumtf.math.pow_19/Pow:z:04tf.math.reduce_sum_19/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_19/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_19/MeanMean"tf.math.reduce_sum_19/Sum:output:0%tf.math.reduce_mean_19/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *ê7
tf.__operators__.add_19/AddV2AddV2$tf.math.reduce_mean_19/Mean:output:0"tf.__operators__.add_19/y:output:0*
T0*
_output_shapes
: Ë
add_loss_19/PartitionedCallPartitionedCall!tf.__operators__.add_19/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_add_loss_19_layer_call_and_return_conditional_losses_104168m
IdentityIdentity"tf.math.reduce_sum_19/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: d

Identity_3Identity$add_loss_19/PartitionedCall:output:1^NoOp*
T0*
_output_shapes
: 
NoOpNoOp!^dense_19/StatefulPartitionedCall#^net_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2D
 dense_19/StatefulPartitionedCall dense_19/StatefulPartitionedCall2H
"net_output/StatefulPartitionedCall"net_output/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_20

¯
F__inference_net_output_layer_call_and_return_conditional_losses_104585

inputs0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
H
,__inference_add_loss_19_layer_call_fn_104564

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_add_loss_19_layer_call_and_return_conditional_losses_104168O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs
Â7
Ï
D__inference_model_49_layer_call_and_return_conditional_losses_104515

inputs9
'dense_19_matmul_readvariableop_resource:;
)net_output_matmul_readvariableop_resource:
identity

identity_1

identity_2

identity_3¢dense_19/MatMul/ReadVariableOp¢ net_output/MatMul/ReadVariableOp
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:*
dtype0{
dense_19/MatMulMatMulinputs&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense_19/ReluReludense_19/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
#dense_19/ActivityRegularizer/SquareSquaredense_19/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
"dense_19/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       
 dense_19/ActivityRegularizer/SumSum'dense_19/ActivityRegularizer/Square:y:0+dense_19/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: g
"dense_19/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *     
 dense_19/ActivityRegularizer/mulMul+dense_19/ActivityRegularizer/mul/x:output:0)dense_19/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: m
"dense_19/ActivityRegularizer/ShapeShapedense_19/Relu:activations:0*
T0*
_output_shapes
:z
0dense_19/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_19/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_19/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_19/ActivityRegularizer/strided_sliceStridedSlice+dense_19/ActivityRegularizer/Shape:output:09dense_19/ActivityRegularizer/strided_slice/stack:output:0;dense_19/ActivityRegularizer/strided_slice/stack_1:output:0;dense_19/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
!dense_19/ActivityRegularizer/CastCast3dense_19/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
$dense_19/ActivityRegularizer/truedivRealDiv$dense_19/ActivityRegularizer/mul:z:0%dense_19/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
 net_output/MatMul/ReadVariableOpReadVariableOp)net_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
net_output/MatMulMatMuldense_19/Relu:activations:0(net_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
net_output/ReluRelunet_output/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%net_output/ActivityRegularizer/SquareSquarenet_output/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
$net_output/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ¤
"net_output/ActivityRegularizer/SumSum)net_output/ActivityRegularizer/Square:y:0-net_output/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: i
$net_output/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ¦
"net_output/ActivityRegularizer/mulMul-net_output/ActivityRegularizer/mul/x:output:0+net_output/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: q
$net_output/ActivityRegularizer/ShapeShapenet_output/Relu:activations:0*
T0*
_output_shapes
:|
2net_output/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4net_output/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4net_output/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
,net_output/ActivityRegularizer/strided_sliceStridedSlice-net_output/ActivityRegularizer/Shape:output:0;net_output/ActivityRegularizer/strided_slice/stack:output:0=net_output/ActivityRegularizer/strided_slice/stack_1:output:0=net_output/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
#net_output/ActivityRegularizer/CastCast5net_output/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: £
&net_output/ActivityRegularizer/truedivRealDiv&net_output/ActivityRegularizer/mul:z:0'net_output/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
tf.math.subtract_19/Sub/yConst*
_output_shapes
:*
dtype0*Q
valueHBF"<D?uÿ>ÍÌÌ=^?ÍÌÌ=B7i>àj>ÍÌÌ=    ÍÌÌ=ÍÌÌ=ÖC>ÍÌÌ=ÍÌÌ=ÍÌÌ=
tf.math.subtract_19/SubSubnet_output/Relu:activations:0"tf.math.subtract_19/Sub/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
tf.math.pow_19/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tf.math.pow_19/PowPowtf.math.subtract_19/Sub:z:0tf.math.pow_19/Pow/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
+tf.math.reduce_sum_19/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
tf.math.reduce_sum_19/SumSumtf.math.pow_19/Pow:z:04tf.math.reduce_sum_19/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
tf.math.reduce_mean_19/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
tf.math.reduce_mean_19/MeanMean"tf.math.reduce_sum_19/Sum:output:0%tf.math.reduce_mean_19/Const:output:0*
T0*
_output_shapes
: ^
tf.__operators__.add_19/yConst*
_output_shapes
: *
dtype0*
valueB
 *ê7
tf.__operators__.add_19/AddV2AddV2$tf.math.reduce_mean_19/Mean:output:0"tf.__operators__.add_19/y:output:0*
T0*
_output_shapes
: m
IdentityIdentity"tf.math.reduce_sum_19/Sum:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh

Identity_1Identity(dense_19/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: j

Identity_2Identity*net_output/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: a

Identity_3Identity!tf.__operators__.add_19/AddV2:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^dense_19/MatMul/ReadVariableOp!^net_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2D
 net_output/MatMul/ReadVariableOp net_output/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

¬
H__inference_dense_19_layer_call_and_return_all_conditional_losses_104542

inputs
unknown:
identity

identity_1¢StatefulPartitionedCallÌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_19_layer_call_and_return_conditional_losses_104121¤
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *9
f4R2
0__inference_dense_19_activity_regularizer_104093o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


$__inference_signature_wrapper_104526
input_20
unknown:
	unknown_0:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinput_20unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_104080k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_20
Í

)__inference_model_49_layer_call_fn_104305
input_20
unknown:
	unknown_0:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinput_20unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_49_layer_call_and_return_conditional_losses_104283k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_20
Í

)__inference_model_49_layer_call_fn_104185
input_20
unknown:
	unknown_0:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinput_20unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: : : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_49_layer_call_and_return_conditional_losses_104175k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_20
¸
s
G__inference_add_loss_19_layer_call_and_return_conditional_losses_104569

inputs
identity

identity_1=
IdentityIdentityinputs*
T0*
_output_shapes
: ?

Identity_1Identityinputs*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¶
serving_default¢
=
input_201
serving_default_input_20:0ÿÿÿÿÿÿÿÿÿE
tf.math.reduce_sum_19,
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÐX

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
±

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
±

kernel
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
(
"	keras_api"
_tf_keras_layer
(
#	keras_api"
_tf_keras_layer
(
$	keras_api"
_tf_keras_layer
(
%	keras_api"
_tf_keras_layer
(
&	keras_api"
_tf_keras_layer
¥
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
w
-iter

.beta_1

/beta_2
	0decay
1learning_ratemPmQvRvS"
	optimizer
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò2ï
)__inference_model_49_layer_call_fn_104185
)__inference_model_49_layer_call_fn_104405
)__inference_model_49_layer_call_fn_104417
)__inference_model_49_layer_call_fn_104305À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_model_49_layer_call_and_return_conditional_losses_104466
D__inference_model_49_layer_call_and_return_conditional_losses_104515
D__inference_model_49_layer_call_and_return_conditional_losses_104346
D__inference_model_49_layer_call_and_return_conditional_losses_104387À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÍBÊ
!__inference__wrapped_model_104080input_20"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
7serving_default"
signature_map
!:2dense_19/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
=activity_regularizer_fn
*&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_19_layer_call_fn_104533¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
H__inference_dense_19_layer_call_and_return_all_conditional_losses_104542¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!2net_output/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
Dactivity_regularizer_fn
*!&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
Õ2Ò
+__inference_net_output_layer_call_fn_104549¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_net_output_layer_call_and_return_all_conditional_losses_104558¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_add_loss_19_layer_call_fn_104564¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_add_loss_19_layer_call_and_return_conditional_losses_104569¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
'
K0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÌBÉ
$__inference_signature_wrapper_104526input_20"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
á2Þ
0__inference_dense_19_activity_regularizer_104093©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
î2ë
D__inference_dense_19_layer_call_and_return_conditional_losses_104577¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ã2à
2__inference_net_output_activity_regularizer_104106©
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢
	
ð2í
F__inference_net_output_layer_call_and_return_conditional_losses_104585¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ltotal
	Mcount
N	variables
O	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
L0
M1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
&:$2Adam/dense_19/kernel/m
(:&2Adam/net_output/kernel/m
&:$2Adam/dense_19/kernel/v
(:&2Adam/net_output/kernel/v¨
!__inference__wrapped_model_1040801¢.
'¢$
"
input_20ÿÿÿÿÿÿÿÿÿ
ª "IªF
D
tf.math.reduce_sum_19+(
tf.math.reduce_sum_19ÿÿÿÿÿÿÿÿÿ
G__inference_add_loss_19_layer_call_and_return_conditional_losses_104569D¢
¢

inputs 
ª ""¢


0 

	
1/0 Y
,__inference_add_loss_19_layer_call_fn_104564)¢
¢

inputs 
ª " Z
0__inference_dense_19_activity_regularizer_104093&¢
¢
	
x
ª " µ
H__inference_dense_19_layer_call_and_return_all_conditional_losses_104542i/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ

	
1/0 £
D__inference_dense_19_layer_call_and_return_conditional_losses_104577[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
)__inference_dense_19_layer_call_fn_104533N/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÕ
D__inference_model_49_layer_call_and_return_conditional_losses_1043469¢6
/¢,
"
input_20ÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H

0ÿÿÿÿÿÿÿÿÿ
-*
	
1/0 
	
1/1 
	
1/2 Õ
D__inference_model_49_layer_call_and_return_conditional_losses_1043879¢6
/¢,
"
input_20ÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H

0ÿÿÿÿÿÿÿÿÿ
-*
	
1/0 
	
1/1 
	
1/2 Ó
D__inference_model_49_layer_call_and_return_conditional_losses_1044667¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H

0ÿÿÿÿÿÿÿÿÿ
-*
	
1/0 
	
1/1 
	
1/2 Ó
D__inference_model_49_layer_call_and_return_conditional_losses_1045157¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H

0ÿÿÿÿÿÿÿÿÿ
-*
	
1/0 
	
1/1 
	
1/2 
)__inference_model_49_layer_call_fn_104185U9¢6
/¢,
"
input_20ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_model_49_layer_call_fn_104305U9¢6
/¢,
"
input_20ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_model_49_layer_call_fn_104405S7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_model_49_layer_call_fn_104417S7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ\
2__inference_net_output_activity_regularizer_104106&¢
¢
	
x
ª " ·
J__inference_net_output_layer_call_and_return_all_conditional_losses_104558i/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "3¢0

0ÿÿÿÿÿÿÿÿÿ

	
1/0 ¥
F__inference_net_output_layer_call_and_return_conditional_losses_104585[/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
+__inference_net_output_layer_call_fn_104549N/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ·
$__inference_signature_wrapper_104526=¢:
¢ 
3ª0
.
input_20"
input_20ÿÿÿÿÿÿÿÿÿ"IªF
D
tf.math.reduce_sum_19+(
tf.math.reduce_sum_19ÿÿÿÿÿÿÿÿÿ