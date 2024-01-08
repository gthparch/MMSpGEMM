#ifndef DYNAMIC_MIN_H
#define DYNAMIC_MIN_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>

template <class T, class Compare>
class DynamicMinMax
{
public:
    DynamicMinMax(const std::vector<T>& data, int m)
    {
        mKeyMap = std::vector<int>(m, -1);
        int i = 0;
        for (auto elem : data) {
            mElements.push_back(elem);
            mKeyMap[elem.key()] = i++;
        }

        make_heap();
    }
    T first() { return *mElements.begin(); }            // WARNING: assumes set not empty
    bool empty() { return mElements.empty(); }

    void update(int id, int new_val)
    {
        if (mKeyMap[id] == -1)
        {
            if (new_val == -1)
                return;

            // inserting a previously deleted element, insert at the end and sift up
            mElements.push_back(T(new_val, id));
            mKeyMap[id] = mElements.size() - 1;
            sift_up(mElements.size() - 1);
            return;
        }

        int p = mKeyMap[id];
        T old_elem = mElements[p];
        if (new_val == -1)
        {
            // deleting the last element
            if (mKeyMap[id] == mElements.size() - 1) {
                mKeyMap[id] = -1;
                mElements.pop_back();
                return;
            }

            // delete this key by swapping with the last element
            mKeyMap[id] = -1;
            mElements[p] = mElements.back();
            mElements.pop_back();
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

    void make_heap()
    {
        int last_nonleaf = (mElements.size() / 2) - 1;
        for (int i = last_nonleaf; i >=0; i--)
            heapify(i);
    }

    void heapify(int node_index)    // sift_down
    {
        int root_index = node_index;
        int left = (node_index * 2) + 1;
        int right = (node_index * 2) + 2;
        if (left < mElements.size() && comp(mElements[left], mElements[root_index]))
            root_index = left;
        if (right < mElements.size() && comp(mElements[right], mElements[root_index]))
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

    void sift_up(int node_index)
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
    std::vector<T> mElements;           // heap
    std::vector<int> mKeyMap;
    Compare comp;
};

#endif // DYNAMIC_MIN_H
