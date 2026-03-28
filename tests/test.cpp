#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <iterator>
#include <vector>

#include "kdtree.hpp"

namespace
{
using Point2 = std::array<float, 2>;
using Tree2 = libkdtree::KDTree<2, Point2>;

/** Point with id so \c find (location) vs \c findExact (==) differ. */
struct PointWithId {
    using value_type = float;
    float data[2]{};
    int id{};

    float operator[](std::size_t i) const { return data[i]; }

    bool operator==(const PointWithId& o) const
    {
        return data[0] == o.data[0] && data[1] == o.data[1] && id == o.id;
    }
};

using TreeTagged = libkdtree::KDTree<2, PointWithId>;

void expectSamePointMultiset(const Tree2& t, std::vector<Point2> expected)
{
    std::vector<Point2> got;
    got.reserve(t.size());
    for (auto it = t.begin(); it != t.end(); ++it) {
        got.push_back(*it);
    }
    std::sort(expected.begin(), expected.end());
    std::sort(got.begin(), got.end());
    EXPECT_EQ(got, expected);
}

struct VisitCounter {
    int count = 0;
    void operator()(Point2 const&) { ++count; }
};
}  // namespace

// --- Lifecycle / capacity ---------------------------------------------------

TEST(KDTree, DefaultEmpty)
{
    Tree2 t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.begin(), t.end());
}

TEST(KDTree, MaxSize)
{
    Tree2 t;
    EXPECT_EQ(t.maxSize(), std::numeric_limits<Tree2::size_type>::max());
}

TEST(KDTree, GetAllocator)
{
    Tree2 t;
    [[maybe_unused]] auto a = t.getAllocator();
}

TEST(KDTree, InsertAndSize)
{
    Tree2 t;
    t.insert(Point2{1.0f, 2.0f});
    t.insert(Point2{3.0f, 4.0f});
    EXPECT_FALSE(t.empty());
    EXPECT_EQ(t.size(), 2u);
}

TEST(KDTree, Clear)
{
    Tree2 t;
    t.insert(Point2{0.0f, 0.0f});
    t.clear();
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
}

TEST(KDTree, RangeConstructorPreservesCount)
{
    std::vector<Point2> pts(50);
    for (int i = 0; i < 50; ++i) {
        pts[static_cast<std::size_t>(i)] = Point2{static_cast<float>(i), static_cast<float>(-i)};
    }
    Tree2 t(pts.begin(), pts.end());
    EXPECT_EQ(t.size(), 50u);
}

TEST(KDTree, CopyConstruct)
{
    Tree2 a;
    a.insert(Point2{1.0f, 1.0f});
    a.insert(Point2{2.0f, 2.0f});
    Tree2 b(a);
    EXPECT_EQ(b.size(), a.size());
    expectSamePointMultiset(b, {{1.f, 1.f}, {2.f, 2.f}});
}

TEST(KDTree, CopyAssign)
{
    Tree2 a;
    a.insert(Point2{1.f, 2.f});
    Tree2 b;
    b.insert(Point2{9.f, 9.f});
    b.insert(Point2{8.f, 8.f});
    a = b;
    EXPECT_EQ(a.size(), 2u);
    expectSamePointMultiset(a, {{8.f, 8.f}, {9.f, 9.f}});
}

TEST(KDTree, CopyAssignSelf)
{
    Tree2 t;
    t.insert(Point2{1.f, 1.f});
    t = t;
    EXPECT_EQ(t.size(), 1u);
}

// --- Accessors / functors ---------------------------------------------------

TEST(KDTree, ValueCompValueAcc)
{
    Tree2 t;
    [[maybe_unused]] auto cmp = t.valueComparator();
    [[maybe_unused]] auto acc = t.valueAccessor();
    const Tree2& ct = t;
    EXPECT_EQ(&ct.valueDistance(), &t.valueDistance());
}

TEST(KDTree, ValueDistanceNonConst)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    auto& d = t.valueDistance();
    (void)d;
    const Tree2& ct = t;
    EXPECT_EQ(&ct.valueDistance(), &t.valueDistance());
}

// --- Iterators --------------------------------------------------------------

TEST(KDTree, ForwardIteration)
{
    Tree2 t;
    t.insert(Point2{1.f, 0.f});
    t.insert(Point2{0.f, 1.f});
    int n = 0;
    for (auto it = t.begin(); it != t.end(); ++it) {
        EXPECT_GE((*it)[0], 0.f);
        ++n;
    }
    EXPECT_EQ(n, 2);
}

TEST(KDTree, ReverseIteration)
{
    Tree2 t;
    t.insert(Point2{1.f, 2.f});
    t.insert(Point2{3.f, 4.f});
    std::vector<Point2> forward;
    for (auto it = t.begin(); it != t.end(); ++it) {
        forward.push_back(*it);
    }
    std::vector<Point2> backward;
    for (auto it = t.rbegin(); it != t.rend(); ++it) {
        backward.push_back(*it);
    }
    std::reverse(backward.begin(), backward.end());
    std::sort(forward.begin(), forward.end());
    std::sort(backward.begin(), backward.end());
    EXPECT_EQ(forward, backward);
}

TEST(KDTree, IteratorArrow)
{
    Tree2 t;
    t.insert(Point2{5.f, 6.f});
    auto it = t.begin();
    ASSERT_NE(it, t.end());
    EXPECT_FLOAT_EQ(it->operator[](0), 5.f);
}

// --- insert -----------------------------------------------------------------

TEST(KDTree, InsertReturnsIterator)
{
    Tree2 t;
    auto it = t.insert(Point2{7.f, 8.f});
    ASSERT_NE(it, t.end());
    EXPECT_FLOAT_EQ((*it)[0], 7.f);
}

TEST(KDTree, InsertWithIgnoredPosition)
{
    Tree2 t;
    t.insert(t.end(), Point2{1.f, 1.f});
    EXPECT_EQ(t.size(), 1u);
}

TEST(KDTree, InsertRange)
{
    Tree2 t;
    std::vector<Point2> v = {{0.f, 0.f}, {1.f, 1.f}, {2.f, 2.f}};
    t.insert(v.begin(), v.end());
    EXPECT_EQ(t.size(), 3u);
}

TEST(KDTree, InsertNCopiesAtIterator)
{
    Tree2 t;
    t.insert(t.begin(), 4u, Point2{3.f, 3.f});
    EXPECT_EQ(t.size(), 4u);
}

TEST(KDTree, InsertIteratorRangeAtPosition)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    std::vector<Point2> more = {{10.f, 10.f}, {11.f, 11.f}};
    t.insert(t.begin(), more.begin(), more.end());
    EXPECT_EQ(t.size(), 3u);
}

// --- erase / find -----------------------------------------------------------

TEST(KDTree, FindMiss)
{
    Tree2 t;
    t.insert(Point2{1.f, 2.f});
    EXPECT_EQ(t.find(Point2{99.f, 99.f}), t.end());
}

TEST(KDTree, FindHit)
{
    Tree2 t;
    t.insert(Point2{4.f, 5.f});
    auto it = t.find(Point2{4.f, 5.f});
    ASSERT_NE(it, t.end());
    EXPECT_FLOAT_EQ((*it)[0], 4.f);
    EXPECT_FLOAT_EQ((*it)[1], 5.f);
}

TEST(KDTree, FindExactDistinctIds)
{
    TreeTagged t;
    PointWithId a{{1.f, 1.f}, 10};
    PointWithId b{{1.f, 1.f}, 20};
    t.insert(a);
    t.insert(b);
    auto ia = t.findExact(a);
    auto ib = t.findExact(b);
    ASSERT_NE(ia, t.end());
    ASSERT_NE(ib, t.end());
    EXPECT_EQ(ia->id, 10);
    EXPECT_EQ(ib->id, 20);
}

TEST(KDTree, EraseByIterator)
{
    Tree2 t;
    t.insert(Point2{1.f, 1.f});
    t.insert(Point2{2.f, 2.f});
    auto it = t.find(Point2{1.f, 1.f});
    ASSERT_NE(it, t.end());
    t.erase(it);
    EXPECT_EQ(t.size(), 1u);
    EXPECT_EQ(t.find(Point2{1.f, 1.f}), t.end());
}

TEST(KDTree, EraseByValueUsesFind)
{
    Tree2 t;
    t.insert(Point2{5.f, 5.f});
    t.erase(Point2{5.f, 5.f});
    EXPECT_TRUE(t.empty());
}

TEST(KDTree, EraseExact)
{
    TreeTagged t;
    PointWithId p{{2.f, 3.f}, 7};
    t.insert(p);
    t.eraseExact(p);
    EXPECT_TRUE(t.empty());
}

// --- Nearest ----------------------------------------------------------------

TEST(KDTree, FindNearest)
{
    std::vector<Point2> pts = {
        {0.0f, 0.0f},
        {10.0f, 0.0f},
        {0.0f, 10.0f},
    };
    Tree2 t(pts.begin(), pts.end());

    Point2 query{0.1f, 0.2f};
    auto [it, dist] = t.findNearest(query);
    ASSERT_NE(it, t.end());
    EXPECT_NEAR((*it)[0], 0.0f, 1e-5f);
    EXPECT_NEAR((*it)[1], 0.0f, 1e-5f);
    EXPECT_LT(dist, 1.0f);
}

TEST(KDTree, FindNearestEmptyReturnsEnd)
{
    Tree2 t;
    auto [it, dist] = t.findNearest(Point2{0.f, 0.f});
    EXPECT_EQ(it, t.end());
    EXPECT_FLOAT_EQ(dist, 0.f);
}

TEST(KDTree, FindNearestWithMaxRadius)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{100.f, 100.f});
    {
        auto [it, d] = t.findNearest(Point2{0.1f, 0.1f}, 0.5f);
        ASSERT_NE(it, t.end());
        EXPECT_LT(d, 0.5f);
    }
    {
        auto [it, d] = t.findNearest(Point2{50.f, 50.f}, 0.001f);
        EXPECT_EQ(it, t.end());
        EXPECT_FLOAT_EQ(d, 0.001f);
    }
}

TEST(KDTree, FindNearestIf)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{10.f, 0.f});
    auto [it, d] = t.findNearestIf(
        Point2{1.f, 0.f}, 100.f,
        [](Point2 const& p) { return p[0] > 5.f; });
    ASSERT_NE(it, t.end());
    EXPECT_GT((*it)[0], 5.f);
    EXPECT_LT(d, 100.f);
}

// --- Range queries (axis-aligned box / L_inf) -------------------------------

TEST(KDTree, CountWithinRangeEmptyTree)
{
    Tree2 t;
    EXPECT_EQ(t.countWithinRange(Point2{0.f, 0.f}, 1.f), 0u);
}

TEST(KDTree, CountWithinRangeCenterRadius)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{1.f, 0.f});
    t.insert(Point2{10.f, 10.f});
    EXPECT_EQ(t.countWithinRange(Point2{0.f, 0.f}, 2.f), 2u);
    EXPECT_EQ(t.countWithinRange(Point2{0.f, 0.f}, 0.1f), 1u);
}

TEST(KDTree, CountWithinRangeRegion)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{2.f, 2.f});
    Tree2::Region reg(Point2{1.f, 1.f}, 1.5f, t.valueAccessor(), t.valueComparator());
    EXPECT_EQ(t.countWithinRange(reg), 2u);
}

TEST(KDTree, VisitWithinRangeCenterRadius)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{5.f, 5.f});
    VisitCounter c;
    VisitCounter out = t.visitWithinRange(Point2{0.f, 0.f}, 1.f, c);
    EXPECT_EQ(out.count, 1);
}

TEST(KDTree, VisitWithinRangeRegion)
{
    Tree2 t;
    t.insert(Point2{1.f, 1.f});
    Tree2::Region reg(Point2{0.f, 0.f}, 2.f, t.valueAccessor(), t.valueComparator());
    VisitCounter c;
    VisitCounter out = t.visitWithinRange(reg, c);
    EXPECT_EQ(out.count, 1);
}

TEST(KDTree, FindWithinRangeEmptyTree)
{
    Tree2 t;
    std::vector<Point2> out;
    t.findWithinRange(Point2{0.f, 0.f}, 1.f, std::back_inserter(out));
    EXPECT_TRUE(out.empty());
}

TEST(KDTree, VisitWithinRangeEmptyTree)
{
    Tree2 t;
    VisitCounter c;
    VisitCounter out = t.visitWithinRange(Point2{0.f, 0.f}, 1.f, c);
    EXPECT_EQ(out.count, 0);
}

TEST(KDTree, FindWithinRangeCenterRadius)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{3.f, 3.f});
    std::vector<Point2> out;
    t.findWithinRange(Point2{0.f, 0.f}, 1.f, std::back_inserter(out));
    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out[0][0], 0.f);
}

TEST(KDTree, FindWithinRangeRegion)
{
    Tree2 t;
    t.insert(Point2{0.f, 0.f});
    t.insert(Point2{10.f, 10.f});
    Tree2::Region reg(Point2{0.f, 0.f}, 0.5f, t.valueAccessor(), t.valueComparator());
    std::vector<Point2> out;
    t.findWithinRange(reg, std::back_inserter(out));
    ASSERT_EQ(out.size(), 1u);
}

// --- optimise / efficient_replace -------------------------------------------

TEST(KDTree, OptimisePreservesElements)
{
    Tree2 t;
    std::vector<Point2> pts = {{3.f, 3.f}, {1.f, 1.f}, {2.f, 2.f}};
    for (auto& p : pts) {
        t.insert(p);
    }
    t.optimize();
    expectSamePointMultiset(t, pts);
    t.checkTreeInvariants();
}

TEST(KDTree, OptimizeAlias)
{
    Tree2 t;
    t.insert(Point2{1.f, 2.f});
    t.optimize();
    EXPECT_EQ(t.size(), 1u);
    t.checkTreeInvariants();
}

TEST(KDTree, EfficientReplaceAndOptimise)
{
    Tree2 t;
    t.insert(Point2{99.f, 99.f});
    std::vector<Point2> v = {{0.f, 0.f}, {1.f, 1.f}};
    t.assignOptimized(v);
    EXPECT_EQ(t.size(), 2u);
    expectSamePointMultiset(t, {{0.f, 0.f}, {1.f, 1.f}});
    t.checkTreeInvariants();
}
