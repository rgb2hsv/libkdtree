#include <array>
#include <gtest/gtest.h>

#include "kdtree.hpp"

TEST(KDTreeSmoke, InsertAndNearest)
{
    libkdtree::KDTree<2, std::array<double, 2>> tree;
    tree.insert({{0.0, 0.0}});
    tree.insert({{10.0, 0.0}});
    auto const [it, dist] = tree.findNearest(std::array<double, 2>{{3.0, 0.0}});
    ASSERT_NE(it, tree.end());
    EXPECT_DOUBLE_EQ((*it)[0], 0.0);
    EXPECT_DOUBLE_EQ((*it)[1], 0.0);
    EXPECT_GT(dist, 0.0);
}
