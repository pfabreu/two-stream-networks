#include <boost/python.hpp>
#include "laplacian_foveation.hpp"


using namespace boost::python;
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(foveate, LaplacianBlending::Foveate, 1, 2)

BOOST_PYTHON_MODULE(yolt_python)
{
    class_< LaplacianBlending >("LaplacianBlending", init<const int,const int,const int,const int,const int >())
      .def("foveate", &LaplacianBlending::Foveate)//args("center"), "foveation function"));
      .def("update_fovea", &LaplacianBlending::CreateFilterPyr);//args("center"), "foveation function"));
}


