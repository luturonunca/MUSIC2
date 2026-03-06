#ifndef CONVEX_HULL_HH
#define CONVEX_HULL_HH

#include <vector>
#include <set>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "logger.hh"

#include <array>


/***** Slow but short convex hull Implementation ******/
/* 
 * Finds the convex hull of a set of data points.
 * Very simple implementation using the O(nh) gift-wrapping algorithm
 * Does not properly take care of degeneracies or round-off problems,
 * in that sense, only 'approximate' convex hull.
 *
 * adapted and expanded from the 'ch3quad' code by Timothy Chan 
 * (https://cs.uwaterloo.ca/~tmchan)
 */

template< typename real_t >
struct convex_hull{
    typedef const real_t *cpr_;
    using vec3_t = std::array<real_t,3>;
    
    size_t npoints_;
    std::vector<int> faceidx_L_, faceidx_U_;
    std::vector<real_t> normals_L_, normals_U_;
    std::vector<real_t> x0_L_, x0_U_;
    vec3_t centroid_;
    real_t volume_;
    vec3_t left_, right_;
    vec3_t anchor_pt_;
    
    inline double turn( cpr_ p, cpr_ q, cpr_ r ) const
    {   return (q[0]-p[0])*(r[1]-p[1]) - (r[0]-p[0])*(q[1]-p[1]);   }
    
    template< bool islower >
    inline double orient( cpr_ p, cpr_ q, cpr_ r, cpr_ s ) const
    {
        if( islower )
            return (q[2]-p[2])*turn(p,r,s) - (r[2]-p[2])*turn(p,q,s) + (s[2]-p[2])*turn(p,q,r);
        
        return (p[2]-q[2])*turn(p,r,s) - (p[2]-r[2])*turn(p,q,s) + (p[2]-s[2])*turn(p,q,r);
    }
    
    inline real_t dot( cpr_ x, cpr_ y ) const 
    {
        return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];
    }
    
    inline real_t det3x3( cpr_ a ) const 
    {
        return (a[0]*(a[4]*a[8]-a[7]*a[5])
                + a[1]*(a[5]*a[6]-a[8]*a[3])
                + a[2]*(a[3]*a[7]-a[6]*a[4]));
    }
    
    void compute_face_normals( cpr_ points )
    {
        normals_L_.assign( faceidx_L_.size(), 0.0 );
        normals_U_.assign( faceidx_U_.size(), 0.0 );
        x0_L_.assign( faceidx_L_.size(), 0.0 );
        x0_U_.assign( faceidx_U_.size(), 0.0 );

        // vertex mean: a guaranteed interior point used to orient all face normals inward
        real_t vmean[3] = {0.0, 0.0, 0.0};
        for( size_t q=0; q<npoints_; ++q )
            for( int p=0; p<3; ++p )
                vmean[p] += points[3*q+p];
        vmean[0] /= npoints_; vmean[1] /= npoints_; vmean[2] /= npoints_;

        #pragma omp parallel for
        for( int i=0; i<(int)faceidx_L_.size()/3; ++i )
        {
            real_t d1[3], d2[3];
            for( int j=0; j<3; ++j )
            {
                x0_L_[3*i+j] = points[ 3*faceidx_L_[3*i+0] + j ];
                d1[j] = points[ 3*faceidx_L_[3*i+1] + j ] - points[ 3*faceidx_L_[3*i+0] + j ];
                d2[j] = points[ 3*faceidx_L_[3*i+2] + j ] - points[ 3*faceidx_L_[3*i+0] + j ];
            }

            normals_L_[3*i+0] = d1[1]*d2[2] - d1[2]*d2[1];
            normals_L_[3*i+1] = d1[2]*d2[0] - d1[0]*d2[2];
            normals_L_[3*i+2] = d1[0]*d2[1] - d1[1]*d2[0];

            // orient toward vertex mean (inward); flip sign if needed
            double tc0 = vmean[0]-x0_L_[3*i+0], tc1 = vmean[1]-x0_L_[3*i+1], tc2 = vmean[2]-x0_L_[3*i+2];
            double dot_c = normals_L_[3*i+0]*tc0 + normals_L_[3*i+1]*tc1 + normals_L_[3*i+2]*tc2;
            double norm_n = sqrt(normals_L_[3*i+0]*normals_L_[3*i+0]+
                                 normals_L_[3*i+1]*normals_L_[3*i+1]+
                                 normals_L_[3*i+2]*normals_L_[3*i+2]);
            if( dot_c < 0.0 ) norm_n = -norm_n;

            normals_L_[3*i+0] /= norm_n;
            normals_L_[3*i+1] /= norm_n;
            normals_L_[3*i+2] /= norm_n;
        }

        #pragma omp parallel for
        for( int i=0; i<(int)faceidx_U_.size()/3; ++i )
        {
            real_t d1[3], d2[3];
            for( int j=0; j<3; ++j )
            {
                x0_U_[3*i+j] = points[ 3*faceidx_U_[3*i+0] + j ];
                d1[j] = points[ 3*faceidx_U_[3*i+1] + j ] - points[ 3*faceidx_U_[3*i+0] + j ];
                d2[j] = points[ 3*faceidx_U_[3*i+2] + j ] - points[ 3*faceidx_U_[3*i+0] + j ];
            }

            normals_U_[3*i+0] = d1[1]*d2[2] - d1[2]*d2[1];
            normals_U_[3*i+1] = d1[2]*d2[0] - d1[0]*d2[2];
            normals_U_[3*i+2] = d1[0]*d2[1] - d1[1]*d2[0];

            // orient toward vertex mean (inward); flip sign if needed
            double tc0 = vmean[0]-x0_U_[3*i+0], tc1 = vmean[1]-x0_U_[3*i+1], tc2 = vmean[2]-x0_U_[3*i+2];
            double dot_c = normals_U_[3*i+0]*tc0 + normals_U_[3*i+1]*tc1 + normals_U_[3*i+2]*tc2;
            double norm_n = sqrt(normals_U_[3*i+0]*normals_U_[3*i+0]+
                                 normals_U_[3*i+1]*normals_U_[3*i+1]+
                                 normals_U_[3*i+2]*normals_U_[3*i+2]);
            if( dot_c < 0.0 ) norm_n = -norm_n;

            normals_U_[3*i+0] /= norm_n;
            normals_U_[3*i+1] /= norm_n;
            normals_U_[3*i+2] /= norm_n;
        }
    }
    
    // Keep only true convex-hull faces: every input point must satisfy
    // dot(q - x0, n) >= -eps.  Interior/spurious faces (from gift-wrapping
    // degeneracies) fail this test for at least one point and are removed.
    void filter_hull_faces( cpr_ points )
    {
        const double eps = 1e-10;

        auto filter = [&]( std::vector<int>& fidx,
                           std::vector<real_t>& normals,
                           std::vector<real_t>& x0s )
        {
            int nf = (int)fidx.size()/3;
            std::vector<int>    fidx_new;
            std::vector<real_t> normals_new, x0_new;
            fidx_new.reserve(fidx.size());
            normals_new.reserve(normals.size());
            x0_new.reserve(x0s.size());

            for( int i=0; i<nf; ++i )
            {
                bool valid = true;
                for( size_t q=0; q<npoints_ && valid; ++q )
                {
                    double d = 0.0;
                    for( int p=0; p<3; ++p )
                        d += normals[3*i+p] * (points[3*q+p] - x0s[3*i+p]);
                    if( d < -eps ) valid = false;
                }
                if( valid )
                {
                    fidx_new.push_back( fidx[3*i+0] );
                    fidx_new.push_back( fidx[3*i+1] );
                    fidx_new.push_back( fidx[3*i+2] );
                    for( int p=0; p<3; ++p ) normals_new.push_back( normals[3*i+p] );
                    for( int p=0; p<3; ++p ) x0_new.push_back( x0s[3*i+p] );
                }
            }

            fidx    = std::move(fidx_new);
            normals = std::move(normals_new);
            x0s     = std::move(x0_new);
        };

        size_t nfL_before = faceidx_L_.size()/3;
        size_t nfU_before = faceidx_U_.size()/3;

        filter( faceidx_L_, normals_L_, x0_L_ );
        filter( faceidx_U_, normals_U_, x0_U_ );

        music::ilog.Print("filter_hull_faces: L %zu->%zu  U %zu->%zu",
            nfL_before, faceidx_L_.size()/3,
            nfU_before, faceidx_U_.size()/3 );
    }

    void compute_center( cpr_ points )
    {
        real_t xc[3] = {0.0,0.0,0.0};
        real_t xcp[3] = {0.0,0.0,0.0};
        
        real_t totvol = 0.0;
        
        for( size_t i=0; i<3*npoints_; ++i )
            xc[i%3] += points[i];
        
        xc[0] /= npoints_;
        xc[1] /= npoints_;
        xc[2] /= npoints_;
        
        
        for( size_t i=0; i<faceidx_L_.size()/3; ++i )
        {
            real_t a[9];
            real_t xct[3] = {xc[0],xc[1],xc[2]};
            
            for( size_t j=0; j<3; ++j )
            {
                for( size_t k=0; k<3; ++k )
                {
                    xct[k] += points[3*faceidx_L_[3*i+j]+k];
                    a[3*j+k] = points[3*faceidx_L_[3*i+j]+k]-xc[k];
                }
            }
            
            xct[0] *= 0.25;
            xct[1] *= 0.25;
            xct[2] *= 0.25;
            
            real_t vol = fabs(det3x3(a))/6.0;
            
            totvol += vol;
            xcp[0] += vol * xct[0];
            xcp[1] += vol * xct[1];
            xcp[2] += vol * xct[2];
        }
        
        for( size_t i=0; i<faceidx_U_.size()/3; ++i )
        {
            real_t a[9];
            real_t xct[3] = {xc[0],xc[1],xc[2]};
            
            for( size_t j=0; j<3; ++j )
            {
                for( size_t k=0; k<3; ++k )
                {
                    xct[k] += points[3*faceidx_U_[3*i+j]+k];
                    a[3*j+k] = points[3*faceidx_U_[3*i+j]+k]-xc[k];
                }
            }
            
            xct[0] *= 0.25;
            xct[1] *= 0.25;
            xct[2] *= 0.25;
            
            real_t vol = fabs(det3x3(a))/6.0;
            
            totvol += vol;
            xcp[0] += vol * xct[0];
            xcp[1] += vol * xct[1];
            xcp[2] += vol * xct[2];
        }
        
        volume_ = totvol;
        
        centroid_[0] = xcp[0] / totvol;
        centroid_[1] = xcp[1] / totvol;
        centroid_[2] = xcp[2] / totvol;
    }
    
    template< bool islower >
    void wrap( cpr_ points, int i, int j, std::vector<int>& idx )
    {
        int k,l,m;
        int h = (int)idx.size()/3;
        
        for( m=0; m<h; ++m )
            if( ( idx[3*m+0]==i && idx[3*m+1]==j ) ||
               ( idx[3*m+1]==i && idx[3*m+2]==j ) ||
               ( idx[3*m+2]==i && idx[3*m+0]==j ) )
                return;
        
        for( k=i, l=0; l < (int)npoints_; ++l )
            if( turn(&points[3*i],&points[3*j],&points[3*l]) < 0 &&
               orient<islower>(&points[3*i],&points[3*j],&points[3*k],&points[3*l]) >=0 )
                k = l;
        
        if( turn(&points[3*i],&points[3*j],&points[3*k]) >= 0 ) return;
        
        idx.push_back( i );
        idx.push_back( j );
        idx.push_back( k );
        
        wrap<islower>( points, k, j, idx );
        wrap<islower>( points, i, k, idx );
    }
    
    template< typename T >
    bool check_point( const T* xp, double dist = 0.0 ) const
    {
        dist *= -1.0;
        
        // take care of possible periodic boundaries
        std::array<T,3> x;
        for( size_t p=0; p<3; ++p )
        {
            real_t d = xp[p] - anchor_pt_[p];
            if( d>0.5 ) x[p] = xp[p]-1.0; else if ( d<-0.5 ) x[p] = xp[p]+1.0; else x[p] = xp[p];
        }

        // check for inside vs. outside
        for( size_t i=0; i<normals_L_.size()/3; ++i )
        {
            std::array<T,3> xp{{x[0]-x0_L_[3*i+0],x[1]-x0_L_[3*i+1],x[2]-x0_L_[3*i+2]}};
            if( dot( &xp[0], &normals_L_[3*i]) < dist ) return false;
        }
        
        for( size_t i=0; i<normals_U_.size()/3; ++i )
        {
            std::array<T,3> xp{{x[0]-x0_U_[3*i+0],x[1]-x0_U_[3*i+1],x[2]-x0_U_[3*i+2]}};
            if( dot( &xp[0], &normals_U_[3*i]) < dist ) return false;
        }
        
        return true;
    }
    
    void expand_vector_from_centroid( real_t* v, double dr  )
    {
        vec3_t dx;
        real_t d = 0.0;
        for( int i=0; i<3; ++i )
        {
            dx[i] = v[i]-centroid_[i];
            d += dx[i]*dx[i];
        }
        d = sqrt(d);
        for( int i=0; i<3; ++i )
            v[i] += dr * dx[i];
    }
    
    void expand( real_t dr )
    {
        for( size_t i=0; i<normals_L_.size(); i+=3 )
            expand_vector_from_centroid( &x0_L_[i], dr );
        for( size_t i=0; i<normals_U_.size(); i+=3 )
            expand_vector_from_centroid( &x0_U_[i], dr );
        
        expand_vector_from_centroid( &left_[0], dr );
        expand_vector_from_centroid( &right_[0], dr );
    }
    
    void get_defining_indices( std::set<int>& unique ) const
    {
        unique.clear();
        
        for( size_t i=0; i<faceidx_L_.size(); ++i )
            unique.insert( faceidx_L_[i] );
        for( size_t i=0; i<faceidx_U_.size(); ++i )
            unique.insert( faceidx_U_[i] );
    }
    
    convex_hull( cpr_ points, size_t npoints, const vec3_t& anchor )
    : npoints_( npoints )
    {
        anchor_pt_[0] = anchor[0];
        anchor_pt_[1] = anchor[1];
        anchor_pt_[2] = anchor[2];
        
        faceidx_L_.reserve(npoints_*3);
        faceidx_U_.reserve(npoints_*3);
        normals_L_.reserve(npoints_*3);
        normals_U_.reserve(npoints_*3);
        x0_L_.reserve(npoints_*3);
        x0_U_.reserve(npoints_*3);
        
        size_t i,j,l;
        for( i=0, l=1; l<npoints_; ++l )
            if( points[3*i] > points[3*l] ) i=l;
        for( j=i, l=0; l<npoints_; ++l )
            if( i!=l && turn(&points[3*i],&points[3*j],&points[3*l]) >= 0 ) j=l;

#ifdef _OPENMP	
        int nt = omp_get_max_threads();
        omp_set_num_threads( std::min(2,omp_get_max_threads()) );
#endif
        
        #pragma omp parallel for
        for( int thread=0; thread<2; ++thread )
        {
            if( thread==0 )
                wrap<true>( points, i, j, faceidx_L_ );
            if( thread==1 )
                wrap<false>( points, i, j, faceidx_U_ );
        }

#ifdef _OPENMP
        omp_set_num_threads(nt);
#endif
        
        compute_face_normals( points );
        filter_hull_faces( points );
        compute_center( points );

        // --- DIAGNOSTIC: verify centroid is inside its own hull ---
        {
            bool centroid_ok = check_point( &centroid_[0] );
            music::ilog.Print("convex_hull diagnostic: centroid check = %s  (nfL=%zu nfU=%zu vol=%.6e)",
                centroid_ok ? "PASS" : "FAIL",
                faceidx_L_.size()/3, faceidx_U_.size()/3, (double)volume_ );
        }
        // --- END DIAGNOSTIC ---

        // finally compute AABB
        left_[0] = left_[1] = left_[2] = 1e30;
        right_[0] = right_[1] = right_[2] = -1e30;
        
        for( size_t q=0; q<npoints_; ++q )
            for( size_t p=0; p<3; ++p )
            {
                if( points[3*q+p] > right_[p] )
                    right_[p] = points[3*q+p];
                if( points[3*q+p] < left_[p] )
                    left_[p] = points[3*q+p];
            }
        
        // make sure left point is inside box
        for( size_t p=0; p<3; ++p )
            if( left_[p] >= 1.0 ){
                left_[p]  -= 1.0; right_[p] -= 1.0;
            }else if( left_[p] < 0.0 ){
                left_[p] += 1.0; right_[p]+= 1.0;
            }
        
    }
    
};


#endif // CONVEX_HULL_HH
