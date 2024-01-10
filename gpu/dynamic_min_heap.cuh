#ifndef DYNAMIC_MIN_H
#define DYNAMIC_MIN_H


#define BITS        14

class Element
{
public:
    __device__ Element() : _m(0) { }
    __device__ Element(unsigned _value, unsigned _id) : _m(((uint32_t)_value << BITS) | _id) { }
    __device__ unsigned key() { return _m & ((1 << BITS) - 1); }
    __device__ unsigned value() { return _m >> BITS; }

    // If columns fit in 18 bits or max nnz in split is less than 14 bits, this could
    // be a uint32_t. Check if it makes a difference in performance...
    uint32_t _m;
};

struct CompareLess
{
    __device__ bool operator()(const Element& a, const Element& b) const
    {
        return a._m < b._m;
    }
};

struct CompareGreater
{
    __device__ bool operator()(const Element& a, const Element& b) const
    {
        return a._m > b._m;
    }
};


template <class T, class Compare>
class DynamicMinMax
{
public:
    __device__ DynamicMinMax(T* data, int size, int *keymap_space, int m)
        : mElements(data), mSize(size), mKeyMap(keymap_space)
    {
        for (int i = 0; i < m; i++)
            mKeyMap[i] = -1;
        for (int i = 0; i < size; i++)
            mKeyMap[mElements[i].key()] = i;

        make_heap();
    }
    __device__ T first() { return mElements[0]; }
    __device__ bool empty() { return mSize == 0; }

    __device__ void update(int id, int new_val)
    {
        if (mKeyMap[id] == -1)
        {
            if (new_val == -1)
                return;

            // inserting a previously deleted element, insert at the end and sift up
            mElements[mSize++] = T(new_val, id);
            mKeyMap[id] = mSize - 1;
            sift_up(mSize - 1);
            return;
        }

        int p = mKeyMap[id];
        T old_elem = mElements[p];
        if (new_val == -1)
        {
            // deleting the last element
            if (mKeyMap[id] == mSize - 1) {
                mKeyMap[id] = -1;
                mSize--;
                return;
            }

            // delete this key by swapping with the last element
            mKeyMap[id] = -1;
            mElements[p] = mElements[mSize-1];
            mSize--;
            mKeyMap[mElements[p].key()] = p;
        }
        else
            mElements[p] = T(new_val, id);

        // check if we should sift up or down
        if (comp(mElements[p], old_elem)) {
            sift_up(p);
        }
        else {
            heapify(p);
        }
    }

    __device__ void make_heap()
    {
        int last_nonleaf = (mSize / 2) - 1;
        for (int i = last_nonleaf; i >=0; i--)
            heapify(i);
    }

    __device__ void heapify(int node_index)    // sift_down
    {
        int root_index = node_index;
        int left = (node_index * 2) + 1;
        int right = (node_index * 2) + 2;
        if (left < mSize && comp(mElements[left], mElements[root_index]))
            root_index = left;
        if (right < mSize && comp(mElements[right], mElements[root_index]))
            root_index = right;

        if (root_index != node_index)
        {
            // swap root_index and node_index
            T tmp = mElements[root_index];
            mElements[root_index] = mElements[node_index];
            mElements[node_index] = tmp;

            // update keymap as well
            mKeyMap[mElements[root_index].key()] = root_index;
            mKeyMap[mElements[node_index].key()] = node_index;

            heapify(root_index);
        }
    }

    __device__ void sift_up(int node_index)
    {
        if (node_index == 0)
            return;
        int parent = (node_index - 1) / 2;

        if (comp(mElements[node_index], mElements[parent]))
        {
            T tmp = mElements[node_index];
            mElements[node_index] = mElements[parent];
            mElements[parent] = tmp;

            mKeyMap[mElements[parent].key()] = parent;
            mKeyMap[mElements[node_index].key()] = node_index;

            sift_up(parent);
        }
    }

protected:
    T *mElements;
    int mSize;
    int *mKeyMap;
    Compare comp;
};

#endif // DYNAMIC_MIN_H
