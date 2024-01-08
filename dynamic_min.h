#ifndef DYNAMIC_MIN_H
#define DYNAMIC_MIN_H

#include <vector>
#include <unordered_map>
#include <set>

template <class T, class Compare>
class DynamicMinMax
{
public:
    DynamicMinMax(const std::vector<T>& data, int m)
    {
        mKeyMap.resize(m);
        for (auto elem : data) {
            mElements.insert(elem);
            mKeyMap[elem.key()] = elem.value();
        }
    }
    T first() { return *mElements.begin(); }            // WARNING: assumes set not empty
    bool empty() { return mElements.empty(); }

    void update(int id, int new_val)
    {
        if (mKeyMap[id] != -1) {
//        auto it = mKeyMap.find(id);
//        if (it != mKeyMap.end()) {
//            mElements.erase(T(it->second, id));
//            mKeyMap.erase(it);
            mElements.erase(T(mKeyMap[id], id));
            mKeyMap[id] = -1;
        }
        if (new_val != -1) {
            mElements.insert(T(new_val, id));
            mKeyMap[id] = new_val;
        }
    }

protected:
    std::set<T, Compare> mElements;
    // TODO: Could be regular array
//    std::unordered_map<int, int>    mKeyMap;        // ID -> value
    std::vector<int> mKeyMap;
};

#endif // DYNAMIC_MIN_H
