var app = angular.module('sample', []);

app.controller('MainCtrl', function($scope, $http) {
  $scope.name = 'World';
  $scope.result=[];


  $http({
    method:'GET',
    url:'people.json'
  }).then(function(result, status){
    $scope.item=result;
  })


// $scope.search= function(val) {
//    $http({
//     method:'GET',
//     url:'people.json'
//   }).then(function(result, status){
//     $scope.item=result;
//   })

//   // fetch data
// }

// $scope.callRestService= function() {
//   $http({method: 'GET', url: 'people.json'}).
//     then(function(data, status, headers, config) {
//          $scope.results.push(data);  //retrieve results and add to existing results
//     })
// }



  $scope.togo=function(data){
  $scope.item2=data;
  }
});