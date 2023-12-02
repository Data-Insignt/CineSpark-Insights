import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

const constantRoutes = [
  {
    path: '/',
    name: 'simple',
    component: (resolve) => require(['@/views/simple'], resolve),
  },
  {
    path: '/als',
    name: 'als',
    component: (resolve) => require(['@/views/als'], resolve),
  },
  {
    path: '/rf',
    name: 'rf',
    component: (resolve) => require(['@/views/rf'], resolve),
  },
  {
    path: '/com',
    name: 'com',
    component: (resolve) => require(['@/views/com'], resolve),
  },
]

export default new Router({
  // mode: 'history', // 去掉url中的#
  scrollBehavior: () => ({ y: 0 }),
  routes: constantRoutes
})
