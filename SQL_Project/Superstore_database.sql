-- Create a database named as superstore_db?
create database superstore_db;

-- Select the  superstore_db; database?
use superstore_db;

-- diplay all records from the superstore table?
select * from superstore_tab;

-- show only product,catetory and total sales?
select Category,Product,Total_Sales from superstore_tab;

-- find all products belonging to the electronics catogory?
select * from superstore_tab where Category = 'Electronics';

-- display orderswhose total sales are greater than 5000?
select superstore_tab where Total_Sales>5000;

-- display all female customers?
select * from  superstore_tab where Gender="Female";

-- show all orders paid using UPI?
select * from superstore_tab where Payment_Method='UPI';

-- display all orders from bengaluru?
select * from superstore_tab where City='Bengaluru';

-- show products with quantity greater than 5?
select * from superstore_tab where Quantity>5;

-- find all preminum customers?
select * from superstore_tab where Customer_Type='Premium';

-- display the first 20 records?
select * from superstore_tab  Limit 20;

-- find the total sales of the store?
select sum(Total_Sales) as Total_sales from superstore_tab;

-- find the average profit>
select avg(profit) as profit from superstore_tab;
 
 -- find the maximum sales value?
 select max(Total_Sales) as Sales from superstore_tab;
 
 -- find the minimum sales value?
 select min(Total_Sales) as sales from superstore_tab
 
 -- count the total number of orders?
 select count(*) as Total_Orders from superstore_tab;
 
 -- find total sales by category?
 select Category, sum(Total_Sales) as Sales from superstore_tab group by Category;
 
 -- find the average rating for each branch?
 select Rating, avg(Branch) as Branch from superstore_tab group by Rating;
 
 -- find the total Quantity sold for each product?
 select Product, sum(Quantity) as Quantity from superstore_tab group by Product;
 
 -- count the number of customers by gender?
 select Gender, count(*)as Total_Customer from superstore_tab group by Gender;
 
 -- display the top 10 highest sales?
 select * from superstore_tab order by Total_Sales desc limit 10;
 
 -- display the lowest 10 profits?
 select * from superstore_tab order by Profit asc limit 10;
 
 -- show products having a discount greater than 20%?
 select * from superstore_tab where  Discount_%  >20;
 
 -- find the orders placed in july?
 select * from superstore_tab where Month='july';
 
 -- display the orders whose rating is above 4.5?
 select * from superstore_tab where Rating> 4.5;
 
 -- show electronics products with profit greater than 1000?
 select * from superstore_tab where Category="Electronics" and Profit>1000;
 
 -- display orders from bengluru and mysuru ?
 select * from superstore_tab where city in ('Bengaluru', 'Mysuru');
 
 -- find products whose names start with'S'?
 select * from superstore_tab where Product like 'S%';
 
 -- find products containing the word 'phone'?
 select * from superstore_tab where Product like '%headPhones%';
 
 -- display orders with total sales between 2000 and 5000?
 select * from superstore_tab where Total_Sales between 2000 and 5000;
 
 -- which city generated the highest sales?
 select City, max(Total_Sales) as Total_sales from superstore_tab group by city order by Total_Sales desc;
 
 -- which category generated the highest profit?
 select Category, max(Profit) as profit from superstore_tab group by Category order by Profit desc;
 
 -- which payment method is used the most?
 select Payment_Method, count(*) as most_payment from superstore_tab group by Payment_Method order by most_payment desc;
 
 -- which branch genrated the maximum revenue?
 select Branch, max(Branch) as revenue from superstore_tab group by Branch order by revenue desc;
 
 -- find monthly sales?
 select Month, sum(Total_Sales) as sales from superstore_tab group by Month order by sales;
 
 -- find the average of each category?
 select Category, avg(Category) as category from superstore_tab group by Category ;
 
 -- find the customer_type contributing maximum sales?
 select Customer_type, max(Total_Sales) as sales  from superstore_tab group by Customer_Type order by sales desc;
 
 -- find the average discount of ecah category?
 select Category, avg(discount) as discount from superstore_tab group by Category;
  
  -- find  the highest-rated products?
  select Product, max(Rating) as rating from superstore_tab group by Product order by rating desc;





