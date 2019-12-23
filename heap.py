# 创建堆时间是O(n)
# pop O(1)
# insert O(logn)
#用于不断调整大根堆
def heap_adjust(n,i,array):
    while 2*i<n:#如果是叶子结点，那么没有子树，不用调整大根堆
        #孩子结点，判断2i+1为左孩子，2i+2为右孩子
        lchild_index=2*i+1
        max_child_index=lchild_index #2i+1
        if lchild_index+1<=n and array[lchild_index+1]>array[lchild_index]:#如果存在右结点
            max_child_index=lchild_index+1
        #和子树的根结点比较
        if array[max_child_index]>array[i]:
            array[i],array[max_child_index]=array[max_child_index],array[i]
            i=max_child_index  #被交换后需要判断是否还需要调整
        else:#如果没有交换数据，那么子树下面不用重构大根堆，退出循环
            break

#构建大根堆
def max_heap(length,array):
    for i in range(length//2-1,-1,-1):#从最后一层第一个非叶子结点开始，依次遍历非叶子结点
        heap_adjust(length-1,i,array)#开始：i代表从第一个非叶子结点
        #print_heap(array)
    return array

#堆排序
def sort(length,array):
    length-=1
    while length>0:
        #堆顶和当前最后一个结点交换，把最大的元素不断的放到数组的最后，只到array[0]才结束
        array[0],array[length]=array[length],array[0]
        length-=1#把最大的数放入数组的最后，放入后，堆调整就不参与了
        if length==1 and array[length]>=array[length-1]:#排完序，结束
            break
        heap_adjust(length,0,array)
        #print_heap(array)
    return array
